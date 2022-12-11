import os
import spacy

class NER:
    def __init__(self):
        os.system("python -m spacy download en_core_web_sm")
        self.ner_classifier=spacy.load("en_core_web_sm")
        
    def get_target_entities(self, words): # NER 활용해서 집단, 사람 관련 Entity만 도출
        # To-Do : Muslim should be detected
        target_entities = []
        for word in words:
            ner_seq=self.ner_classifier(str(word))
            for ent in ner_seq.ents:
                if ((ent.label_ =="NORP") | (ent.label_=="ORG") | (ent.label_=="PERSON")):
                    target_entities.append(str(ent))
        return target_entities # list
    
    def mask_original_text(self, clustered_words, target_entities): # 집단, 사람 관련 Entity를 Masking한 Text 도출
        masked_words = []
        for word in clustered_words:
            if word in target_entities:
                masked_words.append('A person (or people)')
            else:
                masked_words.append(word)
    
        return ' '.join(masked_words)