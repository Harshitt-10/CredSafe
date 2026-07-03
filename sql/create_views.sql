DROP VIEW IF EXISTS bureau_agg;
CREATE VIEW bureau_agg AS
SELECT
    b.SK_ID_CURR,
    COUNT(*) AS bureau_count,
    AVG(b.AMT_CREDIT_SUM) AS bureau_credit_avg,
    MAX(bb.MONTHS_BALANCE) AS bureau_last_month
FROM bureau b
LEFT JOIN bureau_balance bb
ON b.SK_ID_BUREAU = bb.SK_ID_BUREAU
GROUP BY b.SK_ID_CURR;


DROP VIEW IF EXISTS prev_app_agg;
CREATE VIEW prev_app_agg AS
SELECT
    SK_ID_CURR,
    COUNT(*) AS prev_app_count,
    AVG(AMT_APPLICATION) AS prev_app_avg_amt,
    MAX(DAYS_DECISION) AS last_app_days
FROM previous_application
GROUP BY SK_ID_CURR;


DROP VIEW IF EXISTS inst_agg;
CREATE VIEW inst_agg AS
SELECT
    SK_ID_CURR,
    AVG(AMT_PAYMENT / AMT_INSTALMENT) AS pay_to_instal_ratio_avg,
    COUNT(*) AS inst_count
FROM installments_payments
WHERE AMT_INSTALMENT > 0
GROUP BY SK_ID_CURR;


DROP VIEW IF EXISTS cc_agg;
CREATE VIEW cc_agg AS
SELECT
    SK_ID_CURR,
    MAX(MONTHS_BALANCE) AS cc_last_month,
    AVG(AMT_BALANCE) AS cc_avg_balance
FROM credit_card_balance
GROUP BY SK_ID_CURR;


DROP VIEW IF EXISTS pos_agg;
CREATE VIEW pos_agg AS
SELECT
    SK_ID_CURR,
    COUNT(*) AS pos_count,
    MIN(MONTHS_BALANCE) AS pos_first_use
FROM POS_CASH_balance
GROUP BY SK_ID_CURR;