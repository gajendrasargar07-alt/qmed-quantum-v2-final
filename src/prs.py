"""
QMed-Quantum · Polygenic Risk Score (PRS) engine (Synergy 6)
Scores raw DNA genotypes against a published breast-cancer GWAS SNP panel and
returns a relative-risk multiplier + percentile-style banding.

NOTE: allele effect sizes are approximate literature values (Easton et al. 2007,
Turnbull et al. 2010, Michailidou et al. 2017) — demonstrative panel for the
hackathon platform, NOT a clinical-grade polygenic score.
"""
import math

# rsID: (gene/locus, risk allele, per-allele odds ratio, risk-allele frequency)
SNP_PANEL = {
    "rs2981582":  ("FGFR2",  "A", 1.26, 0.38),
    "rs3803662":  ("TOX3",   "T", 1.20, 0.25),
    "rs889312":   ("MAP3K1", "C", 1.13, 0.28),
    "rs3817198":  ("LSP1",   "C", 1.07, 0.30),
    "rs13281615": ("8q24",   "G", 1.08, 0.40),
    "rs13387042": ("2q35",   "A", 1.20, 0.50),
    "rs4973768":  ("SLC4A7", "T", 1.11, 0.47),
    "rs10941679": ("5p12",   "G", 1.19, 0.25),
    "rs2046210":  ("ESR1",   "A", 1.29, 0.35),
    "rs1045485":  ("CASP8",  "C", 0.88, 0.85),   # protective
}

def polygenic_risk_score(genotypes: dict):
    """genotypes: {rsID: 'AG'}. Returns score breakdown."""
    details, log_or_sum, pop_mean = [], 0.0, 0.0
    used = 0
    for rsid, (gene, risk_a, orv, freq) in SNP_PANEL.items():
        g = genotypes.get(rsid)
        if g is None:
            continue
        used += 1
        n_risk = g.count(risk_a)
        beta = math.log(orv)
        log_or_sum += n_risk * beta
        pop_mean += 2 * freq * beta          # expected population contribution
        details.append({"rsid": rsid, "locus": gene, "genotype": g,
                        "risk_allele": risk_a, "risk_copies": n_risk,
                        "odds_ratio": orv,
                        "effect": "↑ risk" if n_risk * beta > 0 else ("↓ protective" if n_risk and beta < 0 else "neutral")})
    if used == 0:
        return None
    relative = math.exp(log_or_sum - pop_mean)
    if relative >= 1.8:
        band, color = "HIGH polygenic risk", "bad"
    elif relative >= 1.15:
        band, color = "ELEVATED polygenic risk", "warn"
    elif relative <= 0.75:
        band, color = "BELOW-AVERAGE polygenic risk", "ok"
    else:
        band, color = "AVERAGE polygenic risk", "info"
    return {"snps_matched": used, "snps_total": len(SNP_PANEL),
            "relative_risk": round(relative, 3), "band": band, "band_color": color,
            "details": details}
