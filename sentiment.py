import pandas as pd

try:
	from textblob import TextBlob
	_HAS_TEXTBLOB = True
except Exception:
	_HAS_TEXTBLOB = False




def _polarity_from_text(text: str) -> float:
	"""Return sentiment polarity in range [-1, 1].

	Uses TextBlob when available; otherwise returns 0.0 as neutral.
	"""
	if _HAS_TEXTBLOB:
		try:
			return TextBlob(text).sentiment.polarity
		except Exception:
			return 0.0
	return 0.0


def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
	"""Adds sentiment polarity score to each message.

	Polarity ranges from -1 (negative) to +1 (positive).
	"""
	df = df.copy()
	scores = []
	for msg in df['message'].astype(str):
		scores.append(_polarity_from_text(msg))
	df['sentiment'] = scores
	return df