import re
import pandas as pd
from collections import Counter
from typing import List

try:
	import emoji
	_HAS_EMOJI = True
except Exception:
	emoji = None
	_HAS_EMOJI = False


URL_RE = re.compile(r'https?://\S+|www\.\S+')
PUNCT_RE = re.compile(r"[\-\_\"'()\[\]{}:;,.!?\\/<>@#$%^&*+=~`|\\]")




def clean_text(text: str) -> str:
	text = text or ''
	text = URL_RE.sub('', text)
	text = text.strip()
	return text




def extract_emojis(s: str) -> List[str]:
	s = s or ''
	if not _HAS_EMOJI:
		return []
	try:
		# emoji v2+: use EMOJI_DATA keys
		emodata = getattr(emoji, 'EMOJI_DATA', None)
		if emodata:
			return [c for c in s if c in emodata]

		# older emoji versions: UNICODE_EMOJI may be mapping or dict with 'en'
		u = getattr(emoji, 'UNICODE_EMOJI', None)
		if u:
			if isinstance(u, dict) and 'en' in u:
				try:
					allowed = set(u['en'].keys()) if isinstance(u['en'], dict) else set(u['en'])
				except Exception:
					allowed = set(u.keys())
			else:
				allowed = set(u.keys()) if isinstance(u, dict) else set(u)
			return [c for c in s if c in allowed]

		# try is_emoji if available
		is_emoji = getattr(emoji, 'is_emoji', None)
		if callable(is_emoji):
			return [c for c in s if is_emoji(c)]
	except Exception:
		pass

	# final fallback: unicode-range regex matching common emoji blocks
	import re
	emoji_pattern = re.compile(
		"[\U0001F600-\U0001F64F"  # emoticons
		"\U0001F300-\U0001F5FF"  # symbols & pictographs
		"\U0001F680-\U0001F6FF"  # transport & map symbols
		"\U0001F1E0-\U0001F1FF"  # flags
		"\U00002700-\U000027BF"  # dingbats
		"\U000024C2-\U0001F251"  # enclosed chars
		"]+", flags=re.UNICODE)
	found = emoji_pattern.findall(s)
	# flatten results into list of characters
	chars = []
	for f in found:
		chars.extend(list(f))
	return chars




def tokenize(text: str) -> List[str]:
	t = clean_text(text)
	# simple tokenization: split on whitespace and remove punctuation
	t = PUNCT_RE.sub(' ', t)
	tokens = [w.strip().lower() for w in t.split() if w.strip()]
	return tokens




def top_n_words(messages, n=20):
	cnt = Counter()
	for m in messages:
		for w in tokenize(m):
			cnt[w] += 1
	return cnt.most_common(n)


def filter_messages_df(df: pd.DataFrame) -> pd.DataFrame:
	"""Return a copy of the DataFrame with unwanted messages removed.

	Filters applied:
	- system/group notification messages (user is None or indicates system)
	- media omitted messages like '<Media omitted>'
	- deleted/removed messages like 'This message was deleted', 'removed you'
	- empty or whitespace-only messages
	"""
	if df is None or df.empty:
		return df

	df = df.copy()
	# ensure message column is string
	df['message'] = df['message'].astype(str)

	# patterns
	media_re = re.compile(r"^(?:<.*media.*omitted.*>|media omitted|<image omitted>|<attached media omitted>|<file omitted>|<media omitted>)$", re.I)
	deleted_re = re.compile(r"(?:this message was deleted|deleted this message|message deleted|removed you|removed|was removed|left the group|you were removed)", re.I)
	system_re = re.compile(r"(?:messages to this chat and calls are now secured|created the group|changed the subject|added|joined using this|left the group)", re.I)

	def _is_unwanted(row):
		user = row.get('user')
		msg = (row.get('message') or '').strip()
		if not msg:
			return True
		# user/system checks
		if user is None:
			return True
		try:
			ul = str(user).lower()
			if 'group_notification' in ul or 'system' in ul:
				return True
		except Exception:
			pass

		# media omitted
		if media_re.search(msg):
			return True

		# deleted/removed/system messages
		if deleted_re.search(msg) or system_re.search(msg):
			return True

		return False

	mask = df.apply(_is_unwanted, axis=1)
	return df.loc[~mask].reset_index(drop=True)