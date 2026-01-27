import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from mlxtend.frequent_patterns import fpgrowth, association_rules
from fpdf import FPDF
import tempfile
import os

# Set Matplotlib backend to Agg to prevent GUI windows
plt.switch_backend('Agg')

# ==============================================================================
# 1. ANALYSIS ENGINE
# ==============================================================================

class AlarmAnalyzer:
    def __init__(self, file_path, min_support, window_size, chatter_cutoff):
        self.file_path = file_path
        self.min_support = min_support
        self.window_size = window_size
        self.chatter_cutoff = chatter_cutoff
        self.df = None
        self.total_alarms = 0
        self.duration_hours = 0
        self.start_time = None
        self.end_time = None

    def load_data(self):
        print("Loading data...")
        try:
            # Try reading col 1 (time) and 2 (alarm) assuming col 0 is index
            self.df = pd.read_csv(self.file_path, usecols=[1, 2], names=['time', 'alarm_text'], header=0)
        except:
            # Fallback
            self.df = pd.read_csv(self.file_path)
            if len(self.df.columns) >= 2:
                self.df.rename(columns={self.df.columns[0]: 'time', self.df.columns[1]: 'alarm_text'}, inplace=True)

        self.df['time'] = pd.to_datetime(self.df['time'])
        self.df['alarm_text'] = self.df['alarm_text'].astype(str)
        self.df = self.df.sort_values('time')
        
        self.total_alarms = len(self.df)
        self.start_time = self.df['time'].min()
        self.end_time = self.df['time'].max()
        
        diff = (self.end_time - self.start_time).total_seconds()
        self.duration_hours = max(diff / 3600, 0.01) # Avoid div by zero
        
        print(f"Loaded {self.total_alarms} alarms. Duration: {self.duration_hours:.2f} hours.")

    def get_most_frequent(self, top_n=10):
        return self.df['alarm_text'].value_counts().head(top_n)

    # --- 1. EEMUA 191 & FLOOD ANALYSIS ---
    def analyze_health(self):
        print("Calculating EEMUA/ISA benchmarks...")
        
        # Average rate
        avg_rate_per_10min = (self.total_alarms / self.duration_hours) / 6
        
        # EEMUA Benchmarks (Alarms per 10 mins)
        if avg_rate_per_10min < 1: status = "ROBUST (Excellent)"
        elif avg_rate_per_10min < 2: status = "STABLE (Good)"
        elif avg_rate_per_10min < 5: status = "REACTIVE (Warning)"
        else: status = "OVERLOADED (Critical)"

        # Flood Analysis (ISA-18.2 > 10 alarms in 10 mins)
        # Resample to 10 min buckets
        floods = self.df.set_index('time').resample('10min')['alarm_text'].count()
        flood_intervals = floods[floods > 10]
        flood_count = len(flood_intervals)
        total_intervals = len(floods)
        pct_flood = (flood_count / total_intervals) * 100 if total_intervals > 0 else 0
        
        return {
            'avg_10min': avg_rate_per_10min,
            'status': status,
            'flood_count': flood_count,
            'flood_pct': pct_flood
        }

    # --- 2. CHATTER ANALYSIS (Kondaveeti et al.) ---
    def analyze_chatter(self):
        print("Analyzing chatter...")
        df_sorted = self.df.sort_values(by=['alarm_text', 'time']).copy()
        df_sorted['prev_time'] = df_sorted.groupby('alarm_text')['time'].shift(1)
        df_sorted['run_length'] = (df_sorted['time'] - df_sorted['prev_time']).dt.total_seconds()
        df_sorted['run_length'] = df_sorted['run_length'].clip(lower=1)
        clean_runs = df_sorted.dropna(subset=['run_length'])

        def calc_psi(group):
            total = len(group)
            if total == 0: return 0.0
            r_counts = group['run_length'].value_counts()
            psi = sum((count/total) * (1/r) for r, count in r_counts.items())
            return psi

        results = (clean_runs.groupby('alarm_text')
                   .apply(calc_psi, include_groups=False)
                   .reset_index(name='chatter_index'))
        
        counts = self.df['alarm_text'].value_counts().reset_index(name='total_occurrences')
        # Fix column names for newer pandas versions
        if counts.shape[1] != 2: counts.columns = ['alarm_text', 'total_occurrences']
        else: counts = counts.rename(columns={counts.columns[0]:'alarm_text', counts.columns[1]:'total_occurrences'})

        results = results.merge(counts, on='alarm_text')
        results['status'] = results['status'] = results['chatter_index'].apply(
            lambda x: 'CRITICAL' if x >= self.chatter_cutoff else 'Normal'
        )
        return results.sort_values(by='chatter_index', ascending=False)

    # --- 3. TEMPORAL ANALYSIS (Hourly) ---
    def analyze_temporal(self):
        print("Analyzing temporal distribution...")
        df_temp = self.df.copy()
        df_temp['hour'] = df_temp['time'].dt.hour
        hourly_counts = df_temp['hour'].value_counts().sort_index()
        # Ensure all 24 hours are present
        for h in range(24):
            if h not in hourly_counts:
                hourly_counts[h] = 0
        return hourly_counts.sort_index()

    # --- 4. PATTERN MINING (FP-Growth) ---
    def analyze_patterns(self):
        print("Mining frequent patterns...")
        basket = (self.df.groupby([pd.Grouper(key='time', freq=self.window_size), 'alarm_text'])['alarm_text']
                  .count().unstack().reset_index().fillna(0)
                  .set_index('time'))
        
        basket_sets = (basket > 0)
        basket_sets = basket_sets[basket_sets.sum(axis=1) > 0]

        if basket_sets.empty: return pd.DataFrame(), pd.DataFrame()

        frequent_itemsets = fpgrowth(basket_sets, min_support=self.min_support, use_colnames=True)
        rules = pd.DataFrame()
        if not frequent_itemsets.empty:
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
            
        return frequent_itemsets, rules

    # --- 5. SEQUENCE MINING (Transition Matrix) ---
    def analyze_sequences(self, top_n=15):
        print("Analyzing sequences (Markov Transitions)...")
        # Filter for only top N alarms to keep matrix readable
        top_alarms = self.df['alarm_text'].value_counts().head(top_n).index.tolist()
        df_seq = self.df[self.df['alarm_text'].isin(top_alarms)].sort_values('time').copy()
        
        df_seq['next_alarm'] = df_seq['alarm_text'].shift(-1)
        df_seq = df_seq.dropna()
        
        # Crosstab: Rows=Current, Cols=Next
        transition_matrix = pd.crosstab(df_seq['alarm_text'], df_seq['next_alarm'], normalize='index')
        return transition_matrix

# ==============================================================================
# 2. VISUALIZATION & REPORTING
# ==============================================================================

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Industrial Alarm System Analysis', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, text):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, text)
        self.ln()
        
    def add_image(self, image_path, w=170):
        if os.path.exists(image_path):
            self.image(image_path, x=20, w=w)
            self.ln(5)

    def create_table(self, df):
        self.set_font('Courier', '', 7)
        col_width = self.w / (len(df.columns) + 1)
        row_height = 5
        
        # Header
        self.set_font('Courier', 'B', 7)
        for col in df.columns:
            text = str(col)[:18]
            self.cell(col_width, row_height, text, border=1)
        self.ln(row_height)
        
        # Rows
        self.set_font('Courier', '', 7)
        for row in df.itertuples(index=False):
            for item in row:
                text = str(item)
                text = text.encode('latin-1', 'replace').decode('latin-1')
                if len(text) > 20: text = text[:17] + "..."
                self.cell(col_width, row_height, text, border=1)
            self.ln(row_height)
        self.ln(5)

def generate_charts(analyzer, freq_counts, chatter_df, hourly_counts, transition_matrix, top_n):
    charts = {}
    
    # 1. Frequency Bar Chart
    plt.figure(figsize=(10, 5))
    freq_counts.sort_values().plot(kind='barh', color='#4C72B0')
    plt.title(f'Top {top_n} Frequent Alarms')
    plt.tight_layout()
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    plt.savefig(path)
    plt.close()
    charts['freq'] = path

    # 2. Hourly Distribution
    plt.figure(figsize=(10, 5))
    hourly_counts.plot(kind='bar', color='#55A868', width=0.8)
    plt.title('Alarm Distribution by Hour of Day')
    plt.xlabel('Hour (0-23)')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    plt.savefig(path)
    plt.close()
    charts['hourly'] = path

    # 3. High Density Alarm Plot (HDAP)
    # Only plot top 25 alarms to keep Y-axis readable, sorted by freq
    top_25 = freq_counts.head(25).index
    df_hdap = analyzer.df[analyzer.df['alarm_text'].isin(top_25)].copy()
    
    plt.figure(figsize=(12, 8))
    plt.scatter(df_hdap['time'], df_hdap['alarm_text'], marker='|', alpha=0.6, s=100, color='#C44E52')
    plt.title('High Density Alarm Plot (HDAP) - Top 25 Tags')
    plt.xlabel('Time')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    plt.savefig(path)
    plt.close()
    charts['hdap'] = path

    # 4. Chatter Index
    plt.figure(figsize=(10, 6))
    top_chatter = chatter_df.head(10).sort_values(by='chatter_index', ascending=True)
    plt.barh(top_chatter['alarm_text'], top_chatter['chatter_index'], color='#DD8452')
    plt.axvline(x=0.05, color='red', linestyle='--', label='Critical Threshold (0.05)')
    plt.title('Top 10 Chattering Alarms')
    plt.xlabel('Chatter Index (Alarms/sec)')
    plt.legend()
    plt.tight_layout()
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    plt.savefig(path)
    plt.close()
    charts['chatter'] = path

    # 5. Sequence Heatmap
    if not transition_matrix.empty:
        plt.figure(figsize=(12, 10))
        sns.heatmap(transition_matrix, annot=False, cmap="Blues", linewidths=.5)
        plt.title(f'Alarm Sequence Probability (Top {top_n} Alarms)')
        plt.xlabel('Next Alarm')
        plt.ylabel('Current Alarm')
        plt.tight_layout()
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        plt.savefig(path)
        plt.close()
        charts['seq'] = path
    else:
        charts['seq'] = None

    return charts

# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Industrial Alarm Analysis Tool v2")
    parser.add_argument('csv_path', type=str, help="Path to CSV")
    parser.add_argument('output_name', type=str, help="Output PDF filename")
    parser.add_argument('--top_n', type=int, default=15, help="Top N alarms")
    parser.add_argument('--min_support', type=float, default=0.01, help="FP-Growth min support (e.g. 0.01 for 1%%)")
    parser.add_argument('--window_size', type=str, default='5min', help="Window size (5min, 1H)")
    parser.add_argument('--cutoff', type=float, default=0.05, help="Chatter threshold")

    args = parser.parse_args()

    # --- RUN ANALYSIS ---
    analyzer = AlarmAnalyzer(args.csv_path, args.min_support, args.window_size, args.cutoff)
    analyzer.load_data()
    
    # 1. Basic Stats
    freq_counts = analyzer.get_most_frequent(args.top_n)
    
    # 2. Health & Benchmarks
    health_metrics = analyzer.analyze_health()
    
    # 3. Temporal
    hourly_counts = analyzer.analyze_temporal()
    
    # 4. Chatter
    chatter_df = analyzer.analyze_chatter()
    
    # 5. Patterns
    itemsets, rules = analyzer.analyze_patterns()
    
    # 6. Sequences
    seq_matrix = analyzer.analyze_sequences(top_n=args.top_n)

    # --- GENERATE CHARTS ---
    print("Generating visualizations...")
    charts = generate_charts(analyzer, freq_counts, chatter_df, hourly_counts, seq_matrix, args.top_n)

    # --- BUILD PDF ---
    print("Building Report...")
    pdf = PDFReport()
    pdf.add_page()
    
    # Executive Summary
    pdf.chapter_title("1. Executive Summary & EEMUA Benchmarks")
    pdf.chapter_body(f"Analysis File: {args.csv_path}")
    pdf.chapter_body(f"Time Range: {analyzer.start_time} to {analyzer.end_time}")
    pdf.chapter_body(f"Total Alarms: {analyzer.total_alarms}")
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 10, f"System Health Status: {health_metrics['status']}", 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 5, f"Avg Alarms per 10 mins: {health_metrics['avg_10min']:.2f}", 0, 1)
    pdf.cell(0, 5, f"Flood Intervals (>10 alarms/10min): {health_metrics['flood_count']} ({health_metrics['flood_pct']:.1f}%)", 0, 1)
    pdf.ln(5)

    # Temporal Analysis
    pdf.chapter_title("2. Temporal Analysis & HDAP")
    pdf.chapter_body("Distribution of alarms by hour of day and High Density Alarm Plot.")
    pdf.add_image(charts['hourly'])
    pdf.add_image(charts['hdap'])

    # Top Alarms
    pdf.chapter_title(f"3. Top {args.top_n} Frequent Alarms")
    pdf.add_image(charts['freq'])

    # Chatter
    pdf.chapter_title("4. Alarm Chatter (Kondaveeti et al.)")
    pdf.chapter_body(f"Alarms with Index > {args.cutoff} are considered CRITICAL (Machine-gunning).")
    pdf.add_image(charts['chatter'])
    
    top_chatter_disp = chatter_df[['alarm_text', 'chatter_index', 'status']].head(10)
    top_chatter_disp['chatter_index'] = top_chatter_disp['chatter_index'].round(4)
    pdf.create_table(top_chatter_disp)

    # Patterns
    pdf.chapter_title("5. Co-Occurrence Patterns (FP-Growth)")
    pdf.chapter_body(f"Parameters: Window={args.window_size}, MinSupport={args.min_support}")
    
    if not itemsets.empty:
        itemsets['len'] = itemsets['itemsets'].apply(len)
        large_patterns = itemsets[itemsets['len'] > 1].sort_values(by='support', ascending=False).head(10)
        
        pdf.set_font('Courier', '', 8)
        for i, row in large_patterns.iterrows():
            items = list(row['itemsets'])
            clean_items = [str(x).encode('latin-1', 'replace').decode('latin-1') for x in items]
            items_str = ", ".join(clean_items)
            pdf.multi_cell(0, 5, f"Support {row['support']*100:.1f}% : {items_str}")
            pdf.ln(2)
    else:
        pdf.chapter_body("No frequent patterns found.")

    # Rules
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 10, "Top Association Rules (Cause -> Effect)", 0, 1)
    if not rules.empty:
        top_rules = rules.sort_values(by='lift', ascending=False).head(8)
        pdf.set_font('Courier', '', 8)
        for i, row in top_rules.iterrows():
            ant = ", ".join([str(x).encode('latin-1', 'replace').decode('latin-1') for x in row['antecedents']])
            con = ", ".join([str(x).encode('latin-1', 'replace').decode('latin-1') for x in row['consequents']])
            pdf.multi_cell(0, 4, f"LIFT {row['lift']:.2f} | {ant} -> {con}")
            pdf.ln(2)

    # Sequence Mining
    pdf.chapter_title("6. Sequence Probability (Markov Matrix)")
    pdf.chapter_body("Heatmap showing the probability of 'Next Alarm' given 'Current Alarm'. Darker blue indicates higher probability.")
    if charts['seq']:
        pdf.add_image(charts['seq'])
    else:
        pdf.chapter_body("Not enough data for sequence analysis.")

    try:
        pdf.output(args.output_name)
        print(f"\nSUCCESS: Report generated at {args.output_name}")
    except Exception as e:
        print(f"Error saving PDF: {e}")

    # Cleanup
    for p in charts.values():
        if p and os.path.exists(p): os.remove(p)

if __name__ == "__main__":
    main()