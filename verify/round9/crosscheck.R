suppressMessages(library(QCA))
setwd("/home/claude/mixtriad/verify/round9")

# B1: 校准对照(同锚点, Ragin 直接法 logistic, idm=0.95)
raw <- read.csv("calib_raw.csv")$x
r_mem <- calibrate(raw, type = "fuzzy", thresholds = c(10, 50, 90), logistic = TRUE, idm = 0.95)
write.csv(data.frame(r_mem = r_mem), "calib_r.csv", row.names = FALSE)

# B2: 示例语料 — pof 测度对照
d <- read.csv("demo_memberships.csv")
sink("r_pof.txt")
print(pof("creator_reach*account_topic_focus -> Y", data = d, relation = "sufficiency"))
print(pof("creator_reach*emotional_intensity -> Y", data = d, relation = "sufficiency"))
print(pof("creator_reach -> Y", data = d, relation = "sufficiency"))
print(pof("~account_topic_focus -> Y", data = d, relation = "sufficiency"))
sink()

# B3: 真值表 + 最小化(保守/简约)
tt <- truthTable(d, outcome = "Y", conditions = "creator_reach, account_topic_focus, emotional_intensity, video_duration",
                 incl.cut = 0.80, n.cut = 2, pri.cut = 0.70, complete = FALSE, show.cases = FALSE)
write.csv(tt$tt, "r_truthtable.csv", row.names = FALSE)
cons <- minimize(tt, details = TRUE)
pars <- minimize(tt, include = "?", details = TRUE)
sink("r_solutions.txt")
cat("== CONSERVATIVE ==\n"); print(cons)
cat("== PARSIMONIOUS ==\n"); print(pars)
sink()

# B4: Lipset 模糊集基准(包内置 LF)
data(LF)
write.csv(LF, "LF.csv", row.names = FALSE)
ttl <- truthTable(LF, outcome = "SURV", incl.cut = 0.8, n.cut = 1)
write.csv(ttl$tt, "r_LF_truthtable.csv", row.names = FALSE)
lc <- minimize(ttl, details = TRUE)
lp <- minimize(ttl, include = "?", details = TRUE)
sink("r_LF_solutions.txt"); cat("== LF CONSERVATIVE ==\n"); print(lc); cat("== LF PARSIMONIOUS ==\n"); print(lp); sink()
cat("R side done\n")
