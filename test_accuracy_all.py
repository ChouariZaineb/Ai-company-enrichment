import json
from src.accuracy import score_all_companies

results = score_all_companies()
print(json.dumps(results, indent=2))