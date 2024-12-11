from transformers import (
    BertConfig, BertForTokenClassification, BertTokenizer, # for KcBERT
    ElectraConfig, ElectraTokenizer, ElectraForTokenClassification,
    AutoConfig, AutoTokenizer, AutoModelForTokenClassification, 
)

def get_model(model_name="koelectra-v3"):
    ModelConfig = None
    Tokenizer = None
    Model = None
    pretrain_name = None
    if model_name == "kcbert":
        ModelConfig = BertConfig
        Tokenizer   = BertTokenizer
        Model       = BertForTokenClassification
        pretrain_name = "beomi/kcbert-base"
    elif model_name == "koelectra-v3" or model_name == "koelectra":
        ModelConfig = ElectraConfig
        Tokenizer   = ElectraTokenizer
        Model       = ElectraForTokenClassification
        if model_name == "koelectra-v3":
            pretrain_name = "monologg/koelectra-base-v3-discriminator"
        elif model_name == "koelectra":
            pretrain_name = "monologg/koelectra-base-discriminator"
    elif model_name == "kobert":
        ModelConfig = AutoConfig
        Tokenizer   = AutoTokenizer
        Model       = AutoModelForTokenClassification
        pretrain_name = "monologg/kobert"
    return ModelConfig, Tokenizer, Model, pretrain_name