# 38_estimator_ablation_dm_ipw_dr.R
# Purpose:
#   Build the "DM vs IPW vs DR" estimator ablation table and the
#   "clip list stability" table for the role-level action set
#   (contest / leverage / backup / hold), using the locked artifacts.
#
# Run (from project root):
#   Rscript Code/38_estimator_ablation_dm_ipw_dr.R
#
# Outputs (written to artifacts/):
#   - estimator_ablation_role_v3.csv
#   - estimator_rank_stability_role_v3.csv
#   - estimator_ablation_role_rows_v3.csv  (row-level diagnostics; optional but helpful)

suppressWarnings(suppressMessages({
  library(tidyverse)
  library(janitor)
  library(glmnet)
  library(xgboost)
  library(readr)
}))

cat("[38] Starting estimator ablation (DM/IPW/DR) for role-only action set\n")

set.seed(2038L)

# ---------------------------
# Paths
# ---------------------------
DIR <- list(artifacts = "artifacts", code = "Code")

PATH <- list(
  lev2            = file.path(DIR$artifacts, "v2_model04_leverage_dataset.csv"),
  elig            = file.path(DIR$artifacts, "eligibility_v3.csv"),
  aff             = file.path(DIR$artifacts, "affordances_v3.csv"),
  dirrec          = file.path(DIR$artifacts, "direction_recommendations_v3.csv"),
  impact_model_rds= file.path(DIR$artifacts, "impact_model_v3.rds"),

  out_ablation    = file.path(DIR$artifacts, "estimator_ablation_role_v3.csv"),
  out_stability   = file.path(DIR$artifacts, "estimator_rank_stability_role_v3.csv"),
  out_rows        = file.path(DIR$artifacts, "estimator_ablation_role_rows_v3.csv")
)

for (nm in c("lev2","elig","aff","dirrec","impact_model_rds")) {
  if (!file.exists(PATH[[nm]])) stop("[38] Missing required artifact: ", PATH[[nm]])
}

# ---------------------------
# Load artifacts (mirror 35)
# ---------------------------
lev2 <- readr::read_csv(PATH$lev2, show_col_types = FALSE) %>%
  clean_names() %>%
  mutate(game_id = as.character(game_id),
         play_id = as.integer(play_id),
         nfl_id  = as.integer(nfl_id),
         split   = tolower(as.character(split)))

elig <- readr::read_csv(PATH$elig, show_col_types = FALSE) %>%
  clean_names() %>%
  mutate(game_id = as.character(game_id),
         play_id = as.integer(play_id),
         nfl_id  = as.integer(nfl_id),
         split   = tolower(as.character(split)))

aff <- readr::read_csv(PATH$aff, show_col_types = FALSE) %>%
  clean_names() %>%
  mutate(game_id = as.character(game_id),
         play_id = as.integer(play_id),
         nfl_id  = as.integer(nfl_id),
         split   = tolower(as.character(split)))

dirrec <- readr::read_csv(PATH$dirrec, show_col_types = FALSE) %>%
  clean_names() %>%
  mutate(game_id = as.character(game_id),
         play_id = as.integer(play_id),
         nfl_id  = as.integer(nfl_id),
         split   = tolower(as.character(split)))

impact_model <- readRDS(PATH$impact_model_rds)
if (is.null(impact_model$outcome_model) || is.null(impact_model$propensity_model)) {
  stop("[38] impact_model_v3.rds is missing outcome_model and/or propensity_model.")
}

action_levels <- impact_model$action_levels
if (is.null(action_levels)) action_levels <- c("contest","leverage","backup","hold")
K <- length(action_levels)

# ---------------------------
# Rebuild base and action_obs (mirror 35)
# ---------------------------
base <- lev2 %>%
  left_join(
    elig %>% select(
      game_id, play_id, nfl_id, split,
      contest_primary, contest_eligible, leverage_eligible, backup_eligible, hold
    ),
    by = c("game_id","play_id","nfl_id","split")
  ) %>%
  left_join(
    aff %>% select(
      game_id, play_id, nfl_id, split,
      r, loi, bv, sep_throw, zone,
      ev_leverage, ev_chase, ev_backup
    ),
    by = c("game_id","play_id","nfl_id","split")
  ) %>%
  mutate(zone = as.character(zone)) %>%
  left_join(
    dirrec %>% select(
      game_id, play_id, nfl_id, split,
      theta_star_deg, theta_star_abs_deg,
      delta_ev_hat, theta_sd_deg,
      backup_point_x, backup_point_y
    ),
    by = c("game_id","play_id","nfl_id","split")
  ) %>%
  mutate(
    contest_primary   = dplyr::coalesce(as.logical(contest_primary), FALSE),
    contest_eligible  = dplyr::coalesce(as.logical(contest_eligible), FALSE),
    leverage_eligible = dplyr::coalesce(as.logical(leverage_eligible), FALSE),
    backup_eligible   = dplyr::coalesce(as.logical(backup_eligible), FALSE),
    hold              = dplyr::coalesce(as.logical(hold), FALSE),
    action_obs = case_when(
      contest_primary ~ "contest",
      !contest_primary & leverage_eligible ~ "leverage",
      !contest_primary & !leverage_eligible & backup_eligible ~ "backup",
      TRUE ~ "hold"
    ),
    action_obs = factor(action_obs, levels = action_levels)
  )

# Filter (mirror 35 and 34)
df <- base %>%
  filter(is.finite(epa),
         !is.na(action_obs)) %>%
  filter(
    is.finite(d_start),
    is.finite(d_min),
    is.finite(d_arrival),
    is.finite(p_reach_start),
    is.finite(p_reach_max)
  ) %>%
  filter(is.finite(r))

cat("[38] Rows after filtering: ", nrow(df), "\n", sep = "")

# ---------------------------
# Feature list (must match 34/35)
# ---------------------------
feat_vars <- c(
  "d_start","p_reach_start","p_reach_max","d_min","d_arrival",
  "sep_throw","r","loi","bv",
  "ext_down","ext_distance","ext_ydstogo",
  "ext_quarter","ext_game_seconds_remaining",
  "ext_score_differential","ext_yardline_100",
  "ext_shotgun","ext_no_huddle",
  "p03_primary","p03_1p0"
)

missing_feats <- setdiff(feat_vars, names(df))
if (length(missing_feats)) {
  warning("[38] Missing feature columns in df (dropping): ", paste(missing_feats, collapse = ", "))
  feat_vars <- intersect(feat_vars, names(df))
}

# ---------------------------
# 1) Outcome model predictions q_hat(s,a) for all actions (DM)
# ---------------------------
xgb_fit    <- impact_model$outcome_model
feat_train <- impact_model$feature_names_out

n_df <- nrow(df)
n_actions <- length(action_levels)

scenarios <- df %>%
  mutate(row_id = dplyr::row_number()) %>%
  { .[rep(seq_len(n_df), each = n_actions), , drop = FALSE] } %>%
  mutate(scenario_action = factor(rep(action_levels, times = n_df), levels = action_levels))

out_df_scen <- scenarios %>%
  mutate(zone = factor(zone),
         action = factor(scenario_action, levels = action_levels)) %>%
  select(all_of(c("action","zone", feat_vars)))

x_scen <- model.matrix(~ . - 1, data = out_df_scen)

# Align columns with training
extra_cols <- setdiff(colnames(x_scen), feat_train)
if (length(extra_cols)) {
  x_scen <- x_scen[, setdiff(colnames(x_scen), extra_cols), drop = FALSE]
}
missing_cols <- setdiff(feat_train, colnames(x_scen))
if (length(missing_cols)) {
  for (nm in missing_cols) {
    x_scen <- cbind(x_scen, rep(0, nrow(x_scen)))
    colnames(x_scen)[ncol(x_scen)] <- nm
  }
}
x_scen <- x_scen[, feat_train, drop = FALSE]

dmat_scen <- xgboost::xgb.DMatrix(data = x_scen)
epa_hat_vec <- as.numeric(predict(xgb_fit, newdata = dmat_scen))
scenarios$epa_hat <- epa_hat_vec

role_wide <- scenarios %>%
  transmute(
    game_id, play_id, nfl_id, split, row_id,
    scenario_action = as.character(scenario_action),
    epa_hat = epa_hat
  ) %>%
  tidyr::pivot_wider(
    names_from  = scenario_action,
    values_from = epa_hat,
    names_prefix = "epa_hat_"
  )

df_scored <- df %>%
  mutate(row_id = dplyr::row_number()) %>%
  left_join(role_wide, by = c("game_id","play_id","nfl_id","split","row_id"))

required_epa_cols <- paste0("epa_hat_", action_levels)
missing_epa_cols <- setdiff(required_epa_cols, names(df_scored))
if (length(missing_epa_cols)) stop("[38] Missing EPA prediction columns: ", paste(missing_epa_cols, collapse = ", "))

df_scored <- df_scored %>%
  mutate(
    epa_hat_contest  = .data[[paste0("epa_hat_", action_levels[1])]],
    epa_hat_leverage = .data[[paste0("epa_hat_", action_levels[2])]],
    epa_hat_backup   = .data[[paste0("epa_hat_", action_levels[3])]],
    epa_hat_hold     = .data[[paste0("epa_hat_", action_levels[4])]]
  ) %>%
  mutate(
    epa_hat_obs = case_when(
      action_obs == "contest"  ~ epa_hat_contest,
      action_obs == "leverage" ~ epa_hat_leverage,
      action_obs == "backup"   ~ epa_hat_backup,
      action_obs == "hold"     ~ epa_hat_hold,
      TRUE ~ NA_real_
    ),
    ev_def_contest  = -epa_hat_contest,
    ev_def_leverage = -epa_hat_leverage,
    ev_def_backup   = -epa_hat_backup,
    ev_def_hold     = -epa_hat_hold,
    ev_def_obs_dm   = -epa_hat_obs
  ) %>%
  mutate(
    ev_def_contest_elig  = if_else(contest_eligible, ev_def_contest, NA_real_),
    ev_def_leverage_elig = if_else(leverage_eligible, ev_def_leverage, NA_real_),
    ev_def_backup_elig   = if_else(backup_eligible, ev_def_backup, NA_real_),
    ev_def_hold_elig     = ev_def_hold,  # hold always available
    ev_def_opt_role_dm = pmax(ev_def_contest_elig, ev_def_leverage_elig,
                              ev_def_backup_elig, ev_def_hold_elig, na.rm = TRUE)
  ) %>%
  mutate(
    action_opt_role = case_when(
      is.finite(ev_def_contest_elig)  & ev_def_contest_elig  == ev_def_opt_role_dm ~ "contest",
      is.finite(ev_def_leverage_elig) & ev_def_leverage_elig == ev_def_opt_role_dm ~ "leverage",
      is.finite(ev_def_backup_elig)   & ev_def_backup_elig   == ev_def_opt_role_dm ~ "backup",
      TRUE ~ "hold"
    ),
    action_opt_role = factor(action_opt_role, levels = action_levels),
    epa_hat_opt_role = case_when(
      action_opt_role == "contest"  ~ epa_hat_contest,
      action_opt_role == "leverage" ~ epa_hat_leverage,
      action_opt_role == "backup"   ~ epa_hat_backup,
      TRUE ~ epa_hat_hold
    ),
    ev_def_opt_role_dm = -epa_hat_opt_role,
    dqs_role_dm = ev_def_opt_role_dm - ev_def_obs_dm
  )

# ---------------------------
# 2) Propensity model μ(a|s): p_obs with uniform mixing (for IPW/DR)
# ---------------------------
prop_fit   <- impact_model$propensity_model
feat_prop  <- impact_model$feature_names_prop
ALPHA_MIX  <- impact_model$propensity_alpha_mix

prop_df <- df_scored %>%
  mutate(zone = factor(zone)) %>%
  select(all_of(c("zone", feat_vars)))

x_prop <- model.matrix(~ . - 1, data = prop_df)

# Align columns with training propensity feature set
extra_p <- setdiff(colnames(x_prop), feat_prop)
if (length(extra_p)) x_prop <- x_prop[, setdiff(colnames(x_prop), extra_p), drop = FALSE]
missing_p <- setdiff(feat_prop, colnames(x_prop))
if (length(missing_p)) {
  for (nm in missing_p) {
    x_prop <- cbind(x_prop, rep(0, nrow(x_prop)))
    colnames(x_prop)[ncol(x_prop)] <- nm
  }
}
x_prop <- x_prop[, feat_prop, drop = FALSE]

p_hat_arr <- predict(prop_fit, newx = x_prop, type = "response", s = impact_model$propensity_lambda)

# glmnet multinomial returns n x K x 1 array
if (length(dim(p_hat_arr)) == 3) {
  p_hat <- p_hat_arr[,,1, drop = FALSE][,,1]
} else if (is.matrix(p_hat_arr)) {
  p_hat <- p_hat_arr
} else {
  stop("[38] Unexpected shape for propensity predictions.")
}

# Ensure columns are in action_levels order
# glmnet uses the response factor levels as columns.
if (!is.null(colnames(p_hat))) {
  # Reorder or subset to match action_levels if possible
  if (all(action_levels %in% colnames(p_hat))) {
    p_hat <- p_hat[, action_levels, drop = FALSE]
  }
}

p_mix <- (1 - ALPHA_MIX) * p_hat + ALPHA_MIX * (1 / K)
p_mix <- p_mix / rowSums(p_mix)

obs_idx <- match(as.character(df_scored$action_obs), action_levels)
p_obs <- p_mix[cbind(seq_len(nrow(df_scored)), obs_idx)]

df_scored <- df_scored %>%
  mutate(p_obs = p_obs)

# ---------------------------
# 3) IPW / DR policy value for π*(s)=action_opt_role (role-only)
# ---------------------------
is_match_star <- as.character(df_scored$action_obs) == as.character(df_scored$action_opt_role)

# Policy value estimates (defensive EV) for π*
ev_pi_dm <- -df_scored$epa_hat_opt_role
ev_pi_ipw <- -ifelse(is_match_star, df_scored$epa / df_scored$p_obs, 0)
dr_epa_star <- df_scored$epa_hat_opt_role + ifelse(is_match_star, (df_scored$epa - df_scored$epa_hat_opt_role) / df_scored$p_obs, 0)
ev_pi_dr <- -dr_epa_star

# Also compute DR-adjusted observed-action EV (for per-row DR dqs)
dr_epa_obs <- df_scored$epa_hat_obs + (df_scored$epa - df_scored$epa_hat_obs) / df_scored$p_obs
ev_obs_dr <- -dr_epa_obs
ev_obs_dm <- -df_scored$epa_hat_obs

df_scored <- df_scored %>%
  mutate(
    is_match_star = is_match_star,
    dr_epa_obs = dr_epa_obs,
    dr_epa_star = dr_epa_star,
    ev_obs_dr = ev_obs_dr,
    ev_pi_dm = ev_pi_dm,
    ev_pi_ipw = ev_pi_ipw,
    ev_pi_dr = ev_pi_dr,
    dqs_role_dr = ( -dr_epa_star ) - ( -dr_epa_obs )  # EV(opt) - EV(obs)
  )

# ---------------------------
# 4) Bootstrap (clustered by game_id) on TEST split
# ---------------------------
df_test <- df_scored %>% filter(split == "test")
if (nrow(df_test) == 0) stop("[38] No rows with split == 'test' found.")

games <- unique(df_test$game_id)
G <- length(games)
cat("[38] TEST split: ", nrow(df_test), " rows across ", G, " games\n", sep="")

B <- 500L
set.seed(3838L)

boot_stat <- function(d) {
  tibble(
    V_pi_DM  = mean(d$ev_pi_dm, na.rm = TRUE),
    V_pi_IPW = mean(d$ev_pi_ipw, na.rm = TRUE),
    V_pi_DR  = mean(d$ev_pi_dr, na.rm = TRUE),
    Delta_DM = mean(d$dqs_role_dm, na.rm = TRUE),
    Delta_DR = mean(d$dqs_role_dr, na.rm = TRUE)
  )
}

boot_vals <- vector("list", B)
for (b in seq_len(B)) {
  g_b <- sample(games, size = G, replace = TRUE)
  d_b <- df_test %>% filter(game_id %in% g_b)
  boot_vals[[b]] <- boot_stat(d_b)
}

boot_df <- bind_rows(boot_vals)

summ_ci <- function(x) {
  q <- quantile(x, probs = c(0.025, 0.5, 0.975), na.rm = TRUE)
  c(lo = as.numeric(q[1]), med = as.numeric(q[2]), hi = as.numeric(q[3]))
}

point <- boot_stat(df_test)

ci <- tibble(
  split = "test",
  V_pi_DM      = point$V_pi_DM,
  V_pi_DM_lo   = summ_ci(boot_df$V_pi_DM)[1],
  V_pi_DM_hi   = summ_ci(boot_df$V_pi_DM)[3],
  V_pi_IPW     = point$V_pi_IPW,
  V_pi_IPW_lo  = summ_ci(boot_df$V_pi_IPW)[1],
  V_pi_IPW_hi  = summ_ci(boot_df$V_pi_IPW)[3],
  V_pi_DR      = point$V_pi_DR,
  V_pi_DR_lo   = summ_ci(boot_df$V_pi_DR)[1],
  V_pi_DR_hi   = summ_ci(boot_df$V_pi_DR)[3],
  Delta_DM     = point$Delta_DM,
  Delta_DM_lo  = summ_ci(boot_df$Delta_DM)[1],
  Delta_DM_hi  = summ_ci(boot_df$Delta_DM)[3],
  Delta_DR     = point$Delta_DR,
  Delta_DR_lo  = summ_ci(boot_df$Delta_DR)[1],
  Delta_DR_hi  = summ_ci(boot_df$Delta_DR)[3]
)

readr::write_csv(ci, PATH$out_ablation)
cat("[38] Wrote estimator ablation table -> ", PATH$out_ablation, "\n", sep="")

# ---------------------------
# 5) Rank stability (DM vs DR clip lists) on TEST (play-level)
# ---------------------------
# Use clipped DQS (>=0) consistent with triage interpretation
play_dm <- df_test %>%
  mutate(dqs_dm_clip = pmax(dqs_role_dm, 0),
         dqs_dr_clip = pmax(dqs_role_dr, 0)) %>%
  group_by(game_id, play_id) %>%
  summarise(
    dqs_play_dm = max(dqs_dm_clip, na.rm = TRUE),
    dqs_play_dr = max(dqs_dr_clip, na.rm = TRUE),
    .groups = "drop"
  )

spearman <- suppressWarnings(cor(play_dm$dqs_play_dm, play_dm$dqs_play_dr, method = "spearman", use = "complete.obs"))

topk <- function(x, k) {
  ord <- order(x, decreasing = TRUE)
  idx <- ord[seq_len(min(k, length(ord)))]
  idx
}

k <- 25L
idx_dm <- topk(play_dm$dqs_play_dm, k)
idx_dr <- topk(play_dm$dqs_play_dr, k)

keys <- function(df, idx) paste(df$game_id[idx], df$play_id[idx], sep = "_")
set_dm <- unique(keys(play_dm, idx_dm))
set_dr <- unique(keys(play_dm, idx_dr))
overlap_k <- length(intersect(set_dm, set_dr))
overlap_pct <- overlap_k / k

# Tail counts at thresholds
taus <- c(0.02, 0.04, 0.06)
tail_rows <- lapply(taus, function(tau) {
  tibble(
    tau = tau,
    nplays_dm = sum(play_dm$dqs_play_dm >= tau, na.rm = TRUE),
    nplays_dr = sum(play_dm$dqs_play_dr >= tau, na.rm = TRUE)
  )
})
tail_df <- bind_rows(tail_rows)

stability <- tibble(
  split = "test",
  spearman_play_max = spearman,
  topk = k,
  topk_overlap = overlap_k,
  topk_overlap_pct = overlap_pct
)

# Flatten tail counts into wide columns
tail_wide <- tail_df %>%
  mutate(tau_label = gsub("\\.", "_", sprintf("tau%.2f", tau))) %>%
  select(tau_label, nplays_dm, nplays_dr) %>%
  pivot_wider(names_from = tau_label,
              values_from = c(nplays_dm, nplays_dr),
              names_sep = "_")

out_stab <- bind_cols(stability, tail_wide)

readr::write_csv(out_stab, PATH$out_stability)
cat("[38] Wrote rank stability table -> ", PATH$out_stability, "\n", sep="")

# Row-level export (optional)
rows_out <- df_scored %>%
  select(game_id, play_id, nfl_id, split,
         action_obs, action_opt_role,
         p_obs, epa, epa_hat_obs, epa_hat_opt_role,
         dr_epa_obs, dr_epa_star,
         dqs_role_dm, dqs_role_dr) %>%
  mutate(
    dqs_role_dm_clip = pmax(dqs_role_dm, 0),
    dqs_role_dr_clip = pmax(dqs_role_dr, 0)
  )

readr::write_csv(rows_out, PATH$out_rows)
cat("[38] Wrote row-level diagnostics -> ", PATH$out_rows, "\n", sep="")

cat("[38] Done ✅\n")
