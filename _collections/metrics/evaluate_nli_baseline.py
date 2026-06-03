import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk

# Ensure you have the punkt tokenizer for sentence splitting
nltk.download('punkt', quiet=True)

class CoverageScorer:
    def __init__(self, model_name="roberta-large-mnli"):
        """
        Initializes the NLI model for entailment checking.
        'roberta-large-mnli' is a standard for checking if a premise entails a hypothesis.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def check_entailment(self, premise, hypothesis):
        """
        Determines if the premise entails the hypothesis (returns 1 for entailment, 0 otherwise).
        
        """
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # The model returns logits for [Contradiction, Neutral, Entailment]
        # Index 2 corresponds to 'Entailment' in the MNLI dataset
        prediction = torch.softmax(outputs.logits, dim=1).argmax().item()
        return 1 if prediction == 2 else 0

    def calculate_score(self, ground_truth, observation_paragraph):
        """
        Calculates the Coverage Score Cov(σ, C).
        """
        # 1. Tokenize the observation paragraph into atomic sentences (m)
        observation_sentences = nltk.sent_tokenize(observation_paragraph)
        n = len(ground_truth)
        
        covered_claims_count = 0

        # 2. Iterate through each claim in the ground truth (j)
        for claim in ground_truth:
            entailed = False
            # 3. Check if any sentence in the observation (i) entails the claim (j)
            for obs_sentence in observation_sentences:
                if self.check_entailment(obs_sentence, claim):
                    entailed = True
                    break  # Disjunction (OR gate): one entailment is enough 
            
            if entailed:
                covered_claims_count += 1

        # 4. Final Coverage Score: sum(covered) / n
        return covered_claims_count / n if n > 0 else 0.0

# --- Example Usage ---

# Ground Truth (List of atomic claims)
ground_truth_claims = [
    "A mountain with snowy peaks is present.",
    "A stone bridge is present.",
    "A blue river is present.",
    "A large green tree is present.",
    "A dark volcano with orange lava is present.",
    "A red castle is present.",
    "A blue moat surrounds the castle.",
    "A green hedge maze is present.",
    "A gray parking lot is present.",
    "Six cars are present in the parking lot.",
    "A football stadium is present.",
    "A windmill is present.",
    "A fenced graveyard with tombstones is present.",
    "A colorful hot air balloon is present.",
    "A dense patch of trees is present.",
    "A blue stone fountain is present.",
    "A giraffe is present in the enclosure.",
    "An elephant is present in the enclosure.",
    "A lion is present in the enclosure."
  ]

# Observation (A paragraph combining atomic sentences)
observation_data = "A green surface is present. The green surface has a textured pattern. A mountain is present. Bushes are green. A beach ball has red sections. A green surface has yellow. A tombstone is present. A gate has black metal. A fence has wooden rails. A person has a blue shirt. A car has blue paint. A parking lot is present. A volcano has a dark rocky exterior. Lava is orange red. A tractor has black wheels. A wooden beam is present. A castle has red turrets. A path has a stone texture. A windmill is present. A stadium light has a metal pole. A scoreboard has a dark screen. A stadium roof has blue panels. Trees are present. A body of water is present. A hot air balloon has red sections. A giraffe has yellow spots. A lion is present. An elephant is present. A stadium has red seats. A hedge maze is present. A rock formation is present. A stone tower is present. A bridge is present. A brown object has a curved shape. A lighthouse has a red top. A fountain has water. A moat is present. A person is present. The person has a blue shirt. A person has a pink shirt."

scorer = CoverageScorer()
score = scorer.calculate_score(ground_truth_claims, observation_data)

print(f"Coverage Score: {score:.2f}")
# Expected Output Logic: 
# Claim 1: Entailed by first sentence.
# Claim 2: Entailed by second sentence.
# Claim 3: Not entailed.
# Score: 2/3 = 0.67