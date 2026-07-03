SELECT
    a.*,

    ba.bureau_count,
    ba.bureau_credit_avg,
    ba.bureau_last_month,

    pa.prev_app_count,
    pa.prev_app_avg_amt,
    pa.last_app_days,

    ia.pay_to_instal_ratio_avg,
    ia.inst_count,

    cca.cc_last_month,
    cca.cc_avg_balance,

    psa.pos_count,
    psa.pos_first_use

FROM application a

LEFT JOIN bureau_agg ba
ON a.SK_ID_CURR = ba.SK_ID_CURR

LEFT JOIN prev_app_agg pa
ON a.SK_ID_CURR = pa.SK_ID_CURR

LEFT JOIN inst_agg ia
ON a.SK_ID_CURR = ia.SK_ID_CURR

LEFT JOIN cc_agg cca
ON a.SK_ID_CURR = cca.SK_ID_CURR

LEFT JOIN pos_agg psa
ON a.SK_ID_CURR = psa.SK_ID_CURR;