from pathlib import Path
import csv
import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_data(csv_path: Path):
    cardinalities = []
    intelligent = []
    sd_intell = []
    dumb = []
    sd_dumb = []

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cardinalities.append(int(row["cardinality"]))
            intelligent.append(float(row["average intelligent"]))
            dumb.append(float(row["average dumb"]))
            sd_intell.append(float(row["standard deviation intelligent"]))
            sd_dumb.append(float(row["standard deviation dumb"]))

    return cardinalities, intelligent, dumb, sd_intell, sd_dumb

def plotGraph(graphName, filename):
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / (filename + ".csv")
    output_path = base_dir / (filename + ".png")
    plt.rcParams['font.size'] = 17

    cardinalities, intelligent, dumb , sd_intell, sd_dumb = load_data(csv_path)

    plt.figure(figsize=(11, 6))
    # draw single vertical error line through each mean (no caps)
    plt.errorbar(cardinalities, intelligent, yerr=sd_intell, marker="o", markersize=8, markeredgewidth=1.5, label="Average runtime of Intelligent Placement's Physical Plan", capsize=8, elinewidth=4)
    # plt.plot(cardinalities, intelligent, marker="o", label="Average runtime of Intelligent Placement's Physical Plan")
    # plt.plot(cardinalities, dumb, marker="s", label="Average runtime of Naive Placement's Physical Plan")
    plt.errorbar(cardinalities, dumb, yerr=sd_dumb, marker="s", markersize=8, markeredgewidth=1.5, label="Average runtime of Naive Placement's Physical Plan", capsize=8, elinewidth=4)
    plt.xlabel("""value of $n$ for the datasets""")
    plt.ylabel("Average execution time (ms)")
    plt.title(graphName)
    # plt.grid(True, alpha=0.3)
    plt.legend(loc=2)
    # plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved plot to {output_path}")

def plotFilterGraph():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "FilterPDResults.csv"
    output_path = base_dir / "FilterPDResults.png"
    plt.rcParams['font.size'] = 18

    cardinalities, intelligent, dumb, sd_intell, sd_dumb = load_data(csv_path)

    plt.figure(figsize=(11, 6))
    # plt.plot(cardinalities, intelligent, marker="o", label="Average runtime of Filtered Logical's Physical Plan")
    # plt.plot(cardinalities, dumb, marker="s", label="Average runtime of Unfiltered Logical's Physical Plan")
    # single-vertical-line errorbars (no caps)
    plt.errorbar(cardinalities, intelligent, yerr=sd_intell, marker="s", markersize=8, markeredgewidth=1.5, label="Average runtime of Filtered Logical's Physical Plan", capsize=8, elinewidth=4)
    plt.errorbar(cardinalities, dumb, yerr=sd_dumb, marker="s", markersize=8, markeredgewidth=1.5, label="Average runtime of Unfiltered Logical's Physical Plan", capsize=8, elinewidth=4)
    plt.xlabel("""value of $n$ for the datasets""")
    plt.ylabel("Average execution time (ms)")
    plt.title("Execution result of Filter Query")
    # plt.grid(True, alpha=0.3)
    plt.legend(loc=4)
    # plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved plot to {output_path}")

def plotBubbleUpGraph():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "BubbleUpResults.csv"
    output_path = base_dir / "BubbleUpResults.png"
    plt.rcParams['font.size'] = 18

    cardinalities, intelligent, dumb, sd_intell, sd_dumb = load_data(csv_path)
    plt.figure(figsize=(11, 6))

    # draw slightly stronger connecting lines behind so the trend is clearer
    # plt.plot(cardinalities, intelligent, color='tab:blue', linestyle='-', linewidth=1.6, alpha=0.85)
    # plt.plot(cardinalities, dumb, color='tab:orange', linestyle='-', linewidth=1.6, alpha=0.85)

    # draw error bars with no connecting line so the vertical bars stand out
    # use a single vertical error line (caps removed) for clearer mean+SD markers
    plt.errorbar(cardinalities, intelligent, yerr=sd_intell, marker='s', markersize=8, markeredgewidth=1.5, label="Average runtime of Bubble-Up Logical's Physical Plan", capsize=8, elinewidth=4)
    plt.errorbar(cardinalities, dumb, yerr=sd_dumb, marker='s', markersize=8, markeredgewidth=1.5, label="Average runtime Initial Logical's Physical Plan", capsize=8, elinewidth=4)

    plt.xlabel("""value of $x$ for the datasets""")
    plt.ylabel("Average execution time (ms)")
    plt.title("Execution result of Bubble-Up Query")
    # keep the original full y-axis scale (do not zoom)
    plt.legend(loc=4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved plot to {output_path}")


def main():
    # base_dir = Path(__file__).resolve().parent
    # csv_path = base_dir / "treePlottingResultsComplex.csv"
    # output_path = base_dir / "treePlottingResultsComplex.png"

    # cardinalities, intelligent, dumb = load_data(csv_path)

    # plt.figure(figsize=(10, 6))
    # plt.plot(cardinalities, intelligent, marker="o", label="Average time required to run the Physical Plan resulting from the Intelligent Placement")
    # plt.plot(cardinalities, dumb, marker="s", label="Average time required to run the Physical Plan resulting from the Naive Placement")
    # plt.xlabel("""value of $n$ for the datasets""")
    # plt.ylabel("Average execution time (ms)")
    # plt.title("Tree Plotting Results Complex")
    # # plt.grid(True, alpha=0.3)
    # plt.legend()
    # # plt.tight_layout()
    # plt.savefig(output_path, dpi=300)
    # plt.close()

    # print(f"Saved plot to {output_path}")

    plotGraph("Extended Tree query pattern execution result", "treePlottingResultsComplex")
    plotGraph("Simple Tree query pattern execution result", "treePlottingResultsSimple")
    plotGraph("Extended Hybrid query with Linear Pattern and Star Pattern execution result", "linearStarPlottingResultsComplex")
    plotGraph("Simple Hybrid query with Linear Pattern and Star Pattern execution result", "linearStarPlottingResultsSimple")
    plotGraph("Extended Hybrid query with Cycle Pattern and Star Pattern execution result", "cycleStarPlottingResultsComplex")
    plotGraph("Simple Hybrid query with Cycle Pattern and Star Pattern execution result", "cycleStarPlottingResultsSimple")
    plotBubbleUpGraph()
    plotFilterGraph()

    


if __name__ == "__main__":
    main()
