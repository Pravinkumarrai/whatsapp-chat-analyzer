import streamlit as st
import pandas as pd

from src.parser import parse_chat
from src.preprocessing import filter_messages_df, tokenize
import re
from src.analytics import (
    total_messages,
    messages_per_user,
    messages_over_time,
    weekday_heatmap,
    most_common_words,
    emoji_counts,
)
from src.sentiment import analyze_sentiment
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import altair as alt


st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")
st.title("📊 WhatsApp Chat Analyzer")

# Add a unique key to avoid duplicate widget IDs across reruns or multiple imports
uploaded_file = st.file_uploader("Upload WhatsApp chat .txt file", type=["txt"], key='upload_chat')

if uploaded_file is not None:
    # read uploaded bytes and decode
    bytes_data = uploaded_file.getvalue()
    try:
        data = bytes_data.decode("utf-8")
    except Exception:
        data = bytes_data.decode("latin-1")

    raw_df = parse_chat(data)
    # keep a filtered copy for display/analysis but retain raw_df for media/link counts
    df = filter_messages_df(raw_df)

    # prepare users list and selection
    df['user'] = df['user'].fillna('System')
    user_list = sorted(df['user'].unique().tolist())
    user_list.insert(0, 'Overall')

    # Sidebar controls: allow selecting one or more users to filter the analysis
    preview_rows = st.sidebar.slider('Preview rows', 5, 100, 10)
    selected_users = st.sidebar.multiselect(
        'Filter by user(s) (leave empty for Overall)',
        options=[u for u in user_list if u != 'Overall'],
        default=[],
        help='Choose one or more users to include. Leave empty to analyze the whole chat.'
    )

    # filter by the selected users (empty selection => Overall)
    if not selected_users:
        view_df = df.copy()
        view_label = 'Overall'
    else:
        view_df = df[df['user'].isin(selected_users)].copy()
        view_label = ', '.join(selected_users)

    st.header("Chat Preview")
    st.write(f"Showing messages for: **{view_label}** — {len(view_df)} messages")
    st.dataframe(view_df[['datetime', 'user', 'message']].head(preview_rows))

    # Top-level statistics (more detailed)
    st.header("Top-level Statistics")
    # basic temporal & user stats
    first_dt = view_df['datetime'].min() if not view_df.empty else None
    last_dt = view_df['datetime'].max() if not view_df.empty else None
    active_days = (last_dt - first_dt).days + 1 if first_dt is not None and last_dt is not None else 0

    # New metrics row: total messages, total words, total links, total media shared
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

    # total messages (in current filtered view)
    total_msgs = total_messages(view_df)
    with stats_col1:
        st.metric("Total Messages", f"{total_msgs:,}")

    # total words (sum of token lengths)
    total_words = 0
    try:
        for m in view_df['message'].astype(str):
            total_words += len(tokenize(m))
    except Exception:
        total_words = 0
    with stats_col2:
        st.metric("Total Words", f"{total_words:,}")

    # total links (count messages containing a URL) — use same pattern as preprocessing
    url_re = re.compile(r'https?://\S+|www\.\S+', re.I)
    total_links = view_df['message'].astype(str).apply(lambda s: bool(url_re.search(s))).sum()
    with stats_col3:
        st.metric("Total Links", f"{int(total_links):,}")

    # total media shared (count in raw_df matching media patterns, filtered to selected users)
    media_re = re.compile(r"(?:<.*media.*omitted.*|media omitted|<image omitted>|<attached media omitted>|<file omitted>)", re.I)
    # apply to raw data but restricted to selected users if any
    try:
        if not selected_users:
            raw_check_df = raw_df
        else:
            raw_check_df = raw_df[raw_df['user'].isin(selected_users)]
        total_media = raw_check_df['message'].astype(str).apply(lambda s: bool(media_re.search(s))).sum()
    except Exception:
        total_media = 0
    with stats_col4:
        st.metric("Total Media Shared", f"{int(total_media):,}")

    # right-hand small summary: first/last/active days
    col1, col2, _ = st.columns(3)
    with col1:
        if first_dt is not None:
            st.write("First message:", pd.to_datetime(first_dt))
    with col2:
        st.write("Active Days", f"{active_days}")

    # Messages per user
    st.header("Messages per User")
    mp = messages_per_user(view_df).reset_index()

    mp.columns = ['user', 'count']
    mp = mp.sort_values('count', ascending=False).reset_index(drop=True)

    if mp.empty:
        st.info('No messages to show per user for the selected filter.')
    else:
        # Controls: choose how many top users to show on the pie (rest -> Others)
        max_users = min(20, len(mp))
        top_n = st.number_input('Top N users in pie (rest grouped as Others)', min_value=1, max_value=max_users, value=min(5, max_users), step=1, key='top_n_users_pie')

        # build pie data: top N users + Others
        top_n = int(top_n)
        top_df = mp.head(top_n).copy()
        others_sum = int(mp['count'].iloc[top_n:].sum()) if len(mp) > top_n else 0
        pie_df = top_df[['user', 'count']].copy()
        if others_sum > 0:
            pie_df = pd.concat([pie_df, pd.DataFrame([{'user': 'Others', 'count': others_sum}])], ignore_index=True)

        # layout: table on left, pie chart on right
        table_col, chart_col = st.columns([2, 1])
        with table_col:
            st.dataframe(mp.head(20).reset_index(drop=True))

        with chart_col:
            try:
                import plotly.express as px
                fig = px.pie(pie_df, names='user', values='count', title='Messages by user (top + Others)', hover_data=['count'])
                fig.update_traces(textinfo='percent+label')
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                try:
                    pie = alt.Chart(pie_df).mark_arc().encode(
                        theta=alt.Theta('count:Q', title='Count'),
                        color=alt.Color('user:N', legend=None),
                        tooltip=[alt.Tooltip('user:N', title='User'), alt.Tooltip('count:Q', title='Count')]
                    ).properties(height=380)
                    st.altair_chart(pie, use_container_width=True)
                except Exception as e:
                    st.error(f'Could not render messages-per-user pie chart: {e}')

    # Messages over time with controls
    st.header("Messages Over Time")
    freq = st.selectbox('Frequency', ['Daily', 'Weekly', 'Monthly'], index=0)
    freq_map = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M'}
    rule = freq_map[freq]
    series = messages_over_time(view_df, freq=rule)
    if series.empty:
        st.info('No data to plot')
    else:
        df_plot = series.reset_index()
        df_plot.columns = ['datetime', 'count']
        chart_type = st.selectbox('Chart type', ['Line', 'Area', 'Bar'], index=0)
        if chart_type == 'Line':
            chart = alt.Chart(df_plot).mark_line(point=True).encode(
                x=alt.X('datetime:T', title='Date'),
                y=alt.Y('count:Q', title='Messages'),
                tooltip=['datetime:T', 'count:Q']
            ).properties(width=700, height=350)
        elif chart_type == 'Area':
            chart = alt.Chart(df_plot).mark_area(opacity=0.6).encode(
                x=alt.X('datetime:T', title='Date'),
                y=alt.Y('count:Q', title='Messages'),
                tooltip=['datetime:T', 'count:Q']
            ).properties(width=700, height=350)
        else:
            chart = alt.Chart(df_plot).mark_bar().encode(
                x=alt.X('datetime:T', title='Date'),
                y=alt.Y('count:Q', title='Messages'),
                tooltip=['datetime:T', 'count:Q']
            ).properties(width=700, height=350)
        st.altair_chart(chart, use_container_width=True)

    # Consecutive Message Analysis (streaks)
    st.header("Consecutive Message Analysis")
    try:
        rows = view_df.sort_values('datetime').reset_index(drop=True)
        streaks = []
        if rows.empty:
            st.info('No messages to analyze consecutive message streaks.')
        else:
            prev_user = None
            streak_len = 0
            start_dt = None
            end_dt = None
            msgs = []
            for _, row in rows.iterrows():
                u = row.get('user', 'System') if isinstance(row, dict) else row['user'] if 'user' in row.index else row.get('user', 'System')
                # normalize NaN users
                if pd.isna(u):
                    u = 'System'

                if u == prev_user:
                    streak_len += 1
                    end_dt = row['datetime']
                    msgs.append(str(row.get('message', '')) if isinstance(row, dict) else str(row['message']))
                else:
                    if prev_user is not None:
                        streaks.append({'user': prev_user, 'start': start_dt, 'end': end_dt, 'length': streak_len, 'sample': msgs[:3]})
                    prev_user = u
                    streak_len = 1
                    start_dt = row['datetime']
                    end_dt = row['datetime']
                    msgs = [str(row.get('message', '')) if isinstance(row, dict) else str(row['message'])]

            # append last streak
            if prev_user is not None:
                streaks.append({'user': prev_user, 'start': start_dt, 'end': end_dt, 'length': streak_len, 'sample': msgs[:3]})

            df_streaks = pd.DataFrame(streaks)
            if df_streaks.empty:
                st.info('No streaks detected.')
            else:
                # summary metrics
                longest = int(df_streaks['length'].max())
                avg_len = round(df_streaks['length'].mean(), 2)
                median_len = float(df_streaks['length'].median())
                total_streaks = int(len(df_streaks))

                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    st.metric('Longest Streak', f"{longest}")
                with s2:
                    st.metric('Average Streak', f"{avg_len}")
                with s3:
                    st.metric('Median Streak', f"{median_len}")
                with s4:
                    st.metric('Total Streaks', f"{total_streaks}")

                # top longest streaks table
                top_n = st.number_input('Show top N longest streaks', min_value=1, max_value=50, value=10, step=1, key='top_streaks_n')
                df_top = df_streaks.sort_values('length', ascending=False).head(int(top_n)).copy()
                if not df_top.empty:
                    # format start/end as datetimes
                    df_top['start'] = pd.to_datetime(df_top['start'])
                    df_top['end'] = pd.to_datetime(df_top['end'])
                    df_top_display = df_top[['user', 'length', 'start', 'end', 'sample']].copy()
                    # join sample messages into a short preview
                    df_top_display['sample'] = df_top_display['sample'].apply(lambda s: ' | '.join(s) if isinstance(s, (list, tuple)) else str(s))
                    st.dataframe(df_top_display.reset_index(drop=True))

                # charts: distribution of streak lengths and average streak per user
                show_charts = st.checkbox('Show consecutive-message charts', key='show_streak_charts')
                if show_charts:
                    try:
                        import plotly.express as px

                        fig1 = px.histogram(df_streaks, x='length', nbins=30, title='Streak Length Distribution', labels={'length':'Streak length'})
                        fig1.update_layout(height=360)
                        st.plotly_chart(fig1, use_container_width=True)

                        # average streak per user
                        avg_user = df_streaks.groupby('user')['length'].mean().reset_index().sort_values('length', ascending=False)
                        fig2 = px.bar(avg_user.head(20), x='length', y='user', orientation='h', title='Average Streak Length by User', labels={'length':'Avg streak','user':'User'})
                        fig2.update_layout(height=450, margin={'l':140})
                        st.plotly_chart(fig2, use_container_width=True)
                    except Exception:
                        try:
                            # Altair fallbacks
                            hist = alt.Chart(df_streaks).mark_bar().encode(
                                x=alt.X('length:Q', bin=alt.Bin(maxbins=30), title='Streak length'),
                                y=alt.Y('count()', title='Count')
                            ).properties(height=360)
                            st.altair_chart(hist, use_container_width=True)

                            avg_user = df_streaks.groupby('user')['length'].mean().reset_index().sort_values('length', ascending=False)
                            bar = alt.Chart(avg_user.head(20)).mark_bar().encode(
                                x=alt.X('length:Q', title='Avg streak'),
                                y=alt.Y('user:N', sort='-x', title='User')
                            ).properties(height=450)
                            st.altair_chart(bar, use_container_width=True)
                        except Exception as e:
                            st.error(f'Could not render streak charts: {e}')
    except Exception as e:
        st.error(f'Error computing consecutive message analysis: {e}')

    # Heatmap: messages per day, faceted by year, weeks on x-axis, weekdays on y-axis
    st.header("Weekly Activity Heatmap")
    # prepare daily counts
    df_dates = view_df.copy()
    if df_dates.empty:
        st.info('No data for heatmap')
    else:
        df_dates['date'] = pd.to_datetime(df_dates['datetime']).dt.date
        daily = df_dates.groupby('date').size().reset_index(name='count')
        daily['date'] = pd.to_datetime(daily['date'])
        daily['year'] = daily['date'].dt.year
        # ISO week number
        daily['week'] = daily['date'].dt.isocalendar().week
        # weekday as name, Monday..Sunday
        import calendar
        daily['weekday'] = daily['date'].dt.weekday
        daily['weekday_name'] = daily['weekday'].apply(lambda x: calendar.day_name[x])
        # create a week label for x-axis showing the week start (Monday) date
        daily['week_start'] = daily['date'] - pd.to_timedelta(daily['weekday'], unit='d')
        daily['week_label'] = daily['week_start'].dt.strftime('%Y-%m-%d')

        # keep order for weekdays Monday..Sunday
        weekday_order = list(calendar.day_name)

        # build altair heatmap faceted by year
        chart_df = daily[['date', 'year', 'week_label', 'weekday_name', 'count']].copy()
        # Convert week_label to a nominal ordered field by sorting by week_start
        week_order = chart_df[['week_label']].drop_duplicates().join(
            chart_df.groupby('week_label')['date'].min(), on='week_label')
        week_order = week_order.sort_values('date')['week_label'].tolist()

        chart_df['week_label'] = pd.Categorical(chart_df['week_label'], categories=week_order, ordered=True)

        import altair as alt
        heat = alt.Chart(chart_df).mark_rect().encode(
            x=alt.X('week_label:N', title='Week (week start)', sort=week_order),
            y=alt.Y('weekday_name:N', title='Weekday', sort=weekday_order),
            color=alt.Color('count:Q', title='Messages', scale=alt.Scale(scheme='blues')),
            tooltip=[alt.Tooltip('date:T', title='Date'), alt.Tooltip('count:Q', title='Messages')]
        ).properties(width=700, height=180)

        # facet by year for year-over-year comparison
        facet = heat.facet(column='year:N')
        st.altair_chart(facet, use_container_width=True)

    # Most common words and emojis

    st.header("Most Common Words")
    words = most_common_words(view_df, n=20)
    # Normalize into a DataFrame with explicit column names
    df_words = pd.DataFrame()
    try:
        if isinstance(words, list) and len(words) > 0 and all(isinstance(x, (list, tuple)) and len(x) >= 2 for x in words):
            df_words = pd.DataFrame(words)
            df_words = df_words.iloc[:, :2]
            df_words.columns = ['word', 'count']
        else:
            # Attempt generic coercion (e.g., Counter.most_common may already be list)
            df_words = pd.DataFrame(words)
            if df_words.shape[1] >= 2:
                df_words = df_words.iloc[:, :2]
                df_words.columns = ['word', 'count']
            else:
                # fallback: try to convert dict-like
                try:
                    df_words = pd.DataFrame(list(words.items()), columns=['word', 'count'])
                except Exception:
                    df_words = pd.DataFrame()
    except Exception:
        df_words = pd.DataFrame()

    if not df_words.empty:
        st.dataframe(df_words.reset_index(drop=True))
        # Optional visualization for most common words (Plotly if available, else Altair)
        show_words_chart = st.checkbox('Show words chart', key='show_words_chart')
        if show_words_chart:
            try:
                import plotly.express as px
                fig = px.bar(df_words, x='count', y='word', orientation='h', text='count', labels={'count': 'Count', 'word': 'Word'})
                fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'}, margin={'l':120})
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                try:
                    chart_words = alt.Chart(df_words).mark_bar().encode(
                        x=alt.X('count:Q', title='Count'),
                        y=alt.Y('word:N', sort='-x', title='Word'),
                        tooltip=[alt.Tooltip('word:N', title='Word'), alt.Tooltip('count:Q', title='Count')]
                    ).properties(height=400)
                    st.altair_chart(chart_words, use_container_width=True)
                except Exception as e:
                    st.error(f'Could not render words chart: {e}')
    else:
        st.info('No common words found for the selected messages.')

    st.header("Top Emojis")
    emojis = emoji_counts(view_df, n=20)
    # Normalize the result into a DataFrame with explicit column names
    try:
        # If emojis is a Series with emoji as index and counts as values
        if hasattr(emojis, 'reset_index'):
            df_emojis = emojis.reset_index()
            # Ensure two-column format
            if df_emojis.shape[1] == 2:
                df_emojis.columns = ['emoji', 'count']
            else:
                # fallback: coerce to DataFrame
                df_emojis = pd.DataFrame(emojis)
        else:
            df_emojis = pd.DataFrame(emojis)
    except Exception:
        df_emojis = pd.DataFrame({'emoji': [], 'count': []})

    # Try to add a human-friendly emoji name using the `emoji` package if available
    try:
        import emoji as emoji_lib

        def _emoji_name(e):
            try:
                name = emoji_lib.demojize(str(e))
                # demojize returns ":smiling_face:" — strip colons and replace underscores
                return name.strip(':').replace('_', ' ')
            except Exception:
                return ''

    except Exception:
        def _emoji_name(e):
            return ''

    # If the emoji_counts returned a list of tuples, coerce with proper column names
    if isinstance(emojis, list) and len(emojis) > 0 and all(isinstance(x, (list, tuple)) and len(x) >= 2 for x in emojis):
        df_emojis = pd.DataFrame(emojis)
        if df_emojis.shape[1] >= 2:
            df_emojis = df_emojis.iloc[:, :2]
            df_emojis.columns = ['emoji', 'count']

    # Add human-friendly name column
    if not df_emojis.empty and 'emoji' in df_emojis.columns:
        df_emojis['emoji'] = df_emojis['emoji'].astype(str)
        df_emojis['name'] = df_emojis['emoji'].apply(_emoji_name)
        df_emojis = df_emojis[['emoji', 'name', 'count']]
        # Show table and a pie chart side-by-side
        table_col, chart_col = st.columns([2, 1])
        with table_col:
            st.dataframe(df_emojis.reset_index(drop=True))
        with chart_col:
            # ensure emoji glyphs are strings
            try:
                df_emojis['emoji'] = df_emojis['emoji'].astype(str)
            except Exception:
                pass

            try:
                import plotly.express as px
                # Use emoji glyphs as slice labels (no names shown) and keep name in hover
                fig = px.pie(df_emojis, names='emoji', values='count', hover_data=['name', 'count'], title='Emoji Distribution')
                fig.update_traces(textinfo='percent+label')
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                try:
                    pie = alt.Chart(df_emojis).mark_arc().encode(
                        theta=alt.Theta('count:Q', title='Count'),
                        color=alt.Color('emoji:N', legend=None),
                        tooltip=[alt.Tooltip('emoji:N', title='Emoji'), alt.Tooltip('count:Q', title='Count')]
                    ).properties(height=380)
                    st.altair_chart(pie, use_container_width=True)
                except Exception as e:
                    st.error(f'Could not render emoji pie chart: {e}')
    else:
        st.info('No emojis found in the selected messages.')

    
    # Wordcloud
    st.header("Word Cloud")
    wc_text = " ".join(view_df['message'].astype(str).tolist())
    if not wc_text.strip():
        st.info('No message text available to generate a word cloud')
    else:
        try:
            wc = WordCloud(width=800, height=400, background_color='white').generate(wc_text)
            fig = plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            st.image(buf)
            plt.close(fig)
        except Exception as e:
            st.error(f'Could not generate word cloud: {e}')

    # Sentiment
    st.header("Sentiment Analysis")
    sent_df = analyze_sentiment(df)
    if sent_df is None or sent_df.empty:
        st.info('No sentiment data available for the selected messages.')
    else:
        sent_df = sent_df.copy()
        # ensure datetime is parsed and sorted
        sent_df['datetime'] = pd.to_datetime(sent_df['datetime'])
        sent_df = sent_df.sort_values('datetime')

        avg_sent = round(sent_df['sentiment'].mean(), 3)
        st.write("Average sentiment:", avg_sent)

        # Chart options: time series with rolling average, or distribution
        chart_option = st.selectbox('Sentiment chart type', ['Time series (with rolling avg)', 'Distribution'], index=0, key='sent_chart_type')

        if chart_option == 'Time series (with rolling avg)':
            roll_window = st.slider('Rolling window (points)', 1, 200, 20, key='sent_roll_window')
            try:
                import plotly.graph_objects as go

                df_plot = sent_df[['datetime', 'sentiment']].copy()
                df_plot['rolling'] = df_plot['sentiment'].rolling(window=roll_window, min_periods=1).mean()

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_plot['datetime'], y=df_plot['sentiment'], mode='lines+markers', name='Sentiment', line=dict(width=1), marker=dict(size=4)))
                fig.add_trace(go.Scatter(x=df_plot['datetime'], y=df_plot['rolling'], mode='lines', name=f'Rolling ({roll_window})', line=dict(width=3)))
                fig.update_layout(title='Sentiment Over Time', xaxis_title='Date', yaxis_title='Sentiment', height=450, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                try:
                    chart = alt.Chart(sent_df).mark_line(point=True).encode(
                        x=alt.X('datetime:T', title='Date'),
                        y=alt.Y('sentiment:Q', title='Sentiment'),
                        tooltip=[alt.Tooltip('datetime:T', title='Date'), alt.Tooltip('sentiment:Q', title='Sentiment')]
                    ).properties(height=450)
                    st.altair_chart(chart, use_container_width=True)
                except Exception as e:
                    st.error(f'Could not render sentiment time series: {e}')

        else:
            # Distribution view
            try:
                import plotly.express as px
                fig = px.histogram(sent_df, x='sentiment', nbins=50, title='Sentiment Distribution', labels={'sentiment':'Sentiment'})
                fig.update_layout(height=360)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                try:
                    hist = alt.Chart(sent_df).mark_bar().encode(
                        x=alt.X('sentiment:Q', bin=alt.Bin(maxbins=50), title='Sentiment'),
                        y=alt.Y('count()', title='Count'),
                        tooltip=[alt.Tooltip('count()', title='Count')]
                    ).properties(height=360)
                    st.altair_chart(hist, use_container_width=True)
                except Exception as e:
                    st.error(f'Could not render sentiment distribution: {e}')


else:
    st.info(
        """
            ### 📌 Follow these steps to analyze your WhatsApp chat:


            **1. Export your WhatsApp chat:**
            - WhatsApp → Settings → Chats → Chat History → Export Chat
            - Choose **Without Media** for best performance.


            **2. Download the exported `.txt` file`.**


            **3. Upload the file in the uploader section above.**


            **4. Wait for processing**, then explore:
            - Some preview of your chat
            - Top-level chat statistics
            - Messages per user
            - Daily/Weekly/Monthly message timeline
            - Consecutive message analysis
            - Weekly Activity heatmap
            - Common words
            - Top emojis
            - Word cloud
            - Sentiment analysis graph/table for individual messages & Average score


            If something fails, check:
            - File is `.txt`
            - Chat was exported without media
            - File content is not edited manually


            🚀 Enjoy your chat analytics!
            """)
    


