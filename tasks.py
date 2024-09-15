from benchmarks import (
    llm_sentiment_analysis, 
    # llm_headline_classification, llm_ner, llm_unit_classification,
    # llm_relation_classification, llm_multiclass_classification, llm_esg_classification, llm_causal_classification,
    # llm_stock_movement_prediction, llm_credit_scoring, llm_fraud_detection, llm_financial_distress_identification,
    # llm_claim_analysis, llm_numeric_labeling, llm_qa
)

TASKS = {
    'sentiment_analysis': llm_sentiment_analysis,
    # 'headline_classification': llm_headline_classification,
    # 'ner': llm_ner,
    # 'unit_classification': llm_unit_classification,
    # 'relation_classification': llm_relation_classification,
    # 'multiclass_classification': llm_multiclass_classification,
    # 'esg_classification': llm_esg_classification,
    # 'causal_classification': llm_causal_classification,
    # 'stock_movement_prediction': llm_stock_movement_prediction,
    # 'credit_scoring': llm_credit_scoring,
    # 'fraud_detection': llm_fraud_detection,
    # 'financial_distress_identification': llm_financial_distress_identification,
    # 'claim_analysis': llm_claim_analysis,
    # 'numeric_labeling': llm_numeric_labeling,
    # 'question_answering': llm_qa,
}
