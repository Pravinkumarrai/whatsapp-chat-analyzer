import re
from datetime import datetime
import pandas as pd
from dateutil import parser as dateparser


# Common WhatsApp patterns (Android / iOS)
DATETIME_PREFIX_RE = re.compile(r"^(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}),?\s(\d{1,2}:\d{2}(?::\d{2})?)(?:\s[APMapm]{2})?\s?-\s+")


LINE_START_RE = re.compile(r"^(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}),?\s(\d{1,2}:\d{2}(?::\d{2})?)(?:\s[APMapm]{2})?\s-\s")


SENDER_MSG_RE = re.compile(r"^(.+?):\s(.*)$")




import re
import pandas as pd

try:
	import dateparser
	def _parse_date(s: str):
		try:
			return dateparser.parse(s, dayfirst=False, fuzzy=True)
		except Exception:
			# fall through to dateutil fallback below if available
			try:
				from dateutil import parser as _dateutil_parser
				return _dateutil_parser.parse(s, dayfirst=False)
			except Exception:
				return None
except Exception:
	from dateutil import parser as _dateutil_parser
	def _parse_date(s: str):
		try:
			return _dateutil_parser.parse(s, dayfirst=False)
		except Exception:
			return None


# Common WhatsApp patterns (Android / iOS)
LINE_START_RE = re.compile(r"^(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}),?\s(\d{1,2}:\d{2}(?::\d{2})?)(?:\s[APMapm]{2})?\s-\s")
SENDER_MSG_RE = re.compile(r"^(.+?):\s(.*)$")


def _is_new_message(line: str) -> bool:
	return bool(LINE_START_RE.match(line))


def parse_chat(text: str) -> pd.DataFrame:
	"""Parse raw WhatsApp export text and return DataFrame with columns: datetime, user, message

	This parser is tolerant: handles multi-line messages and common Android/iOS formats.
	"""
	lines = text.splitlines()
	messages = []

	cur_dt = None
	cur_user = None
	cur_msg_lines = []

	for line in lines:
		if LINE_START_RE.match(line):
			# flush previous message (if any)
			if cur_msg_lines:
				messages.append({
					"datetime": cur_dt,
					"user": cur_user,
					"message": "\n".join(cur_msg_lines).strip()
				})

			# parse new line: split datetime prefix and rest
			parts = line.split(' - ', 1)
			dt_part = parts[0].strip()
			rest = parts[1] if len(parts) > 1 else ''
			cur_dt = _parse_date(dt_part)

			# extract user and message
			m = SENDER_MSG_RE.match(rest)
			if m:
				cur_user = m.group(1).strip()
				cur_msg_lines = [m.group(2).strip()]
			else:
				cur_user = None
				cur_msg_lines = [rest.strip()]
		else:
			# continuation line
			if cur_msg_lines is not None:
				cur_msg_lines.append(line)

	# flush last message
	if cur_msg_lines:
		messages.append({
			"datetime": cur_dt,
			"user": cur_user,
			"message": "\n".join(cur_msg_lines).strip()
		})

	df = pd.DataFrame(messages)
	# drop rows without message text
	if 'message' in df.columns:
		df = df[df['message'].notna()].copy()
	# ensure datetime column
	if 'datetime' in df.columns:
		df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
	df = df.sort_values('datetime').reset_index(drop=True) if not df.empty else df
	return df


if __name__ == '__main__':
	import sys
	text = open(sys.argv[1], encoding='utf-8', errors='ignore').read()
	df = parse_chat(text)
	print(df.head())
