import pandas as pd
from collections import Counter
from .preprocessing import tokenize, extract_emojis


def total_messages(df: pd.DataFrame) -> int:
	return len(df)




def messages_per_user(df: pd.DataFrame):
	return df['user'].fillna('System').value_counts()




def messages_over_time(df: pd.DataFrame, freq='D'):
	return df.set_index('datetime').resample(freq).size()




def hourly_distribution(df: pd.DataFrame):
	return df['datetime'].dt.hour.value_counts().sort_index()




def weekday_heatmap(df: pd.DataFrame):
	# returns pivot table: weekday (0=Mon) x hour
	df = df.copy()
	df['weekday'] = df['datetime'].dt.weekday
	df['hour'] = df['datetime'].dt.hour
	pivot = df.pivot_table(index='weekday', columns='hour', values='message', aggfunc='count', fill_value=0)
	return pivot




def most_common_words(df: pd.DataFrame, n=20):
	return tokenize_top_n(df['message'].astype(str).tolist(), n)




def tokenize_top_n(messages, n=20):
	cnt = Counter()
	for m in messages:
		for w in tokenize(m):
			cnt[w] += 1
	return cnt.most_common(n)




def emoji_counts(df: pd.DataFrame, n=20):
	cnt = Counter()
	for m in df['message'].astype(str):
		for e in extract_emojis(m):
			cnt[e] += 1
	return cnt.most_common(n)