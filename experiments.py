import csv
import itertools
import pickle
import random
import re
from collections import defaultdict
from datetime import datetime, timedelta
from os.path import realpath, dirname, isfile

import matplotlib
import seaborn as sns

import networkx as nx
import matplotlib.pyplot as plt
import subprocess

GRAPHS = [
    ("results_erdos_renyi_5000_0.002", lambda: nx.erdos_renyi_graph(5000, 0.002), "Erdos Renyi graph (N=5000, P=0.02)"),
    ("results_dense_gnm_1000_25000", lambda: nx.dense_gnm_random_graph(5000, 25000),
     "Dense $G_{n,m}$ ($N=1000$, $M=25000$)"),
    ("results_extended_ba_2000_3", lambda: nx.extended_barabasi_albert_graph(2000, 3, 0.002, 0.002),
     "Extended BA ($N=2000$, $M=3$, $P=0.002$, $Q=0.02$)"),
    ("results_small_graph", lambda: nx.erdos_renyi_graph(850, 0.01), "Small test"),
]
GRAPH = GRAPHS[3]

KS = [1, 7, 14, 21]


def get_graph():
    file_name = f"{GRAPH[0]}.gpickle"

    if isfile(file_name): return pickle.load(open(file_name, "rb"))

    G = GRAPH[1]()
    G = nx.DiGraph([(u, v, {"prob": random.uniform(0.0, 1.0)}) for (u, v) in G.edges() if u < v])
    pickle.dump(G, open(file_name, "wb"), pickle.HIGHEST_PROTOCOL)
    return G


def output_graph_edges_paths(G):
    o = open("./edges.pl", "w+")

    props = nx.get_edge_attributes(G, "prob")
    for (u, v) in G.edges():
        o.write(f"{props[(u, v)]}::dir_edge({u},{v}).\n")

    generations = list(nx.topological_generations(G))
    sources, sinks = generations[0], generations[-1]

    paths = []
    for source, sink in itertools.product(sources, sinks):
        paths.append((source, sink))

    samples = min(50, len(paths))
    return random.sample(paths, samples)


def run_experiment_yap(K, paths):
    kbest, koptimal = [], []
    for (source, sink) in paths:
        kbest.append(f"problog_kbest(path({source},{sink}),{K},Prob,Status).")
        koptimal.append(f"problog_koptimal(path({source},{sink}),{K},Prob).")

    running_container = subprocess.run('docker ps --filter "ancestor=yap:latest" --format "{{.ID}}"', shell=True,
                                       capture_output=True, text=True).stdout
    if len(running_container) == 0:
        running_container = subprocess.run(
            f'docker run --mount type=bind,source={realpath(dirname(__file__))},destination=/opt/scripts -t -d yap:latest',
            shell=True, capture_output=True, text=True).stdout

    running_container = running_container.strip()
    print(f"Running in container {running_container}")

    with open(f"{GRAPH[0]}.csv", "a+") as results:
        for i, (cmd_best, cmd_optimal, (source, sink)) in enumerate(zip(kbest, koptimal, paths)):
            has_failed, to_write = False, ""
            for algo in ["kbest", "koptimal"]:
                command = open("command_yap.pl", "w+")
                command.write(f"""?- [graph_yap].
            ?- {cmd_best if algo == "kbest" else cmd_optimal}""")
                command.flush()

                output = subprocess.run(
                    f'docker exec {running_container} sh -c "time yap -L /opt/scripts/command_yap.pl"',
                    shell=True, capture_output=True, text=True).stderr

                if algo == "kbest":
                    [_, prob, status, time, _] = output.splitlines()
                    if status != "Status = ok": print(output)
                    prob = re.search('Prob = (.+),', prob).group(1)
                else:
                    if len(output.splitlines()) != 4:
                        print(f"OUTPUT FAILED {output} k={K}, algo={algo}, sample={i}")
                        has_failed = True
                        continue
                    [_, prob, time, _] = output.splitlines()
                    prob = re.search('Prob = (.+)', prob).group(1)

                time = re.search('(.+)user (.+)system (.+)elapsed (.+)%CPU', time).group(3)
                print(f"k={K}, algo={algo}, sample={i} ({source},{sink}): {prob}, {time}")
                to_write += f"{algo},{K},{i},{prob},{time}\n"

            if not has_failed: results.write(to_write)
            print("")
            results.flush()


def run_experiment_problog(paths):
    with open(f"{GRAPH[0]}.csv", "a+") as results:
        for i, (source, sink) in enumerate(paths):
            to_write = ""
            for algo in ["sdd", "nnf", "kbest"]:
                command = open("command_problog.pl", "w+")
                command.write(f""":- consult(graph_problog).
                query(path({source},{sink})).""")
                command.flush()

                output = subprocess.run(
                    f'time problog --timeout 10 --knowledge {algo} command_problog.pl',
                    shell=True, capture_output=True, text=True)

                if "Timeout" in output.stdout: continue

                prob = re.search('(.+):\t(.+)\n', output.stdout).group(2).strip()
                time = re.search('(.+)user (.+)system (.+)elapsed (.+)%CPU', output.stderr.splitlines()[2]).group(3)

                print(f"algo={algo}, sample={i} ({source},{sink}): {prob}, {time}")
                to_write += f"{algo}_problog,,{i},{prob},{time}\n"

            results.write(to_write)
            print("")
            results.flush()


def draw_plots():
    graphs = [1, 2]

    def map_graph(graph_i):
        csv_data = csv.reader(open(f"{GRAPHS[graph_i][0]}.csv", "r"))
        return list(filter(lambda d: d[0] == 'kbest' or d[0] == 'koptimal', csv_data))

    csv_data = [(graph_i, map_graph(graph_i)) for graph_i in graphs]

    fig, (
        (ax1_l, ax1_r, ax2_l, ax2_r),
        (ax3_l, ax3_r, ax4_l, ax4_r)
    ) = plt.subplots(2, 4, width_ratios=[3, 1, 3, 1], figsize=(8, 6), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=4 / 72, hspace=0.2,
                                    wspace=0)

    def handle_plot(ax_l, ax_r, data, label, graph_i):
        xs = list(map(lambda d: float(d[3]), filter(lambda d: d[0] == 'kbest', data)))
        ys = list(map(lambda d: float(d[3]), filter(lambda d: d[0] == 'koptimal', data)))

        # Scat
        ax_l.set_xlim(0, 1)
        ax_l.set_ylim(0, 1)
        ax_l.scatter(xs, ys, s=10, label=GRAPHS[graph_i][2])

        ax_l.set_title(label, fontsize=10)
        ax_l.legend().remove()

        # Diff
        diff_data = [defaultdict(lambda: 0) for _ in range(len(KS))]

        for d in data:
            data_i = KS.index(int(d[1]))
            diff_data[data_i][int(d[2])] += (-1 if d[0] == 'kbest' else 1) * float(d[3])

        diff_data = list(map(lambda d: list(d.values()), diff_data))

        xs_strip = [f"$K={KS[i]}$" for i, xs in enumerate(diff_data) for x in xs]
        ys_strip = [x for xs in diff_data for x in xs]
        sns.stripplot(x=xs_strip, y=ys_strip, ax=ax_r, label=GRAPHS[graph_i][2])

        ax_r.yaxis.tick_right()
        ax_r.legend().remove()

    for (graph_i, data) in csv_data:
        handle_plot(ax1_l, ax1_r, list(filter(lambda d: d[1] == '1', data)), "K=1", graph_i)
        handle_plot(ax2_l, ax2_r, list(filter(lambda d: d[1] == '7', data)), "K=7", graph_i)
        handle_plot(ax3_l, ax3_r, list(filter(lambda d: d[1] == '14', data)), "K=14", graph_i)
        handle_plot(ax4_l, ax4_r, list(filter(lambda d: d[1] == '21', data)), "K=21", graph_i)

    matplotlib.rcParams.update({
        "text.usetex": True,
    })

    handles, labels = ax4_r.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.5))

    fig.show()
    fig.savefig(f"probs.png", dpi=450)


def draw_time_increase_plots():
    graphs = [1, 2]

    def map_graph(graph_i):
        csv_data = list(csv.reader(open(f"{GRAPH[0]}.csv", "r")))
        return list(filter(lambda d: d[0] == 'kbest' or d[0] == 'koptimal', csv_data))

    csv_data = [(graph_i, map_graph(graph_i)) for graph_i in graphs]

    for (graph_i, graph_data) in csv_data:
        data = [dict() for _ in range(len(KS))]

        for d in graph_data:
            data_i = KS.index(int(d[1]))

            dur = datetime.strptime(d[4] + '0000', '%M:%S.%f')
            delta = timedelta(minutes=dur.minute, seconds=dur.second, microseconds=dur.microsecond)

            if d[0] == 'kbest':
                data[data_i][int(d[2])] = delta
            else:
                kbest_delta = data[data_i][int(d[2])]
                data[data_i][int(d[2])] = 100 * (
                        delta.total_seconds() - kbest_delta.total_seconds()) / kbest_delta.total_seconds()

        data = list(map(lambda d: list(d.values()), data))

        xs = [f"K={KS[i]}" for i, xs in enumerate(data) for x in xs]
        ys = [x for xs in data for x in xs]
        ax = sns.stripplot(x=xs, y=ys, label=GRAPHS[graph_i][2], legend=None)

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys())

    matplotlib.rcParams.update({
        "text.usetex": True,
    })

    plt.ylabel("Increase in run-time for $k$-Optimal (%)")
    plt.yscale("symlog")
    plt.savefig(f"runtime.png", dpi=450)
    plt.show()


def draw_time_compare_plots():
    csv_data = list(csv.reader(open(f"{GRAPH[0]}.csv", "r")))

    algorithms = set(map(lambda d: d[0], csv_data))
    for algo in algorithms:
        data = list(filter(lambda d: d[0] == algo, csv_data))

        def parse_time(t):
            dur = datetime.strptime(t + '0000', '%M:%S.%f')
            delta = timedelta(minutes=dur.minute, seconds=dur.second, microseconds=dur.microsecond)
            return delta.total_seconds()

        ys = list(map(lambda d: parse_time(d[4]), data))
        sns.stripplot(x=[algo.strip("_problog")] * len(data), y=ys)

    plt.ylabel("Run-time per algorithm (s)")
    plt.savefig(f"runtime_problog.png", dpi=450)
    plt.show()


G = get_graph()

# paths = output_graph_edges_paths(G)
# run_experiment_problog(paths)
# for K in KS:
#     run_experiment_yap(K, paths)

# draw_plots()
# draw_time_increase_plots()
draw_time_compare_plots()