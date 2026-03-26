import json
import os
import threading
from datetime import datetime

# Load .env file from project root
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from flask import Flask, render_template, jsonify, request, send_file
from game.game import Game
from game.player import MODELS

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"))

# Store benchmark results
benchmark_results = []
game_in_progress = False
current_game_log = []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/models")
def get_models():
    return jsonify({"models": MODELS})


@app.route("/api/start_game", methods=["POST"])
def start_game():
    global game_in_progress, current_game_log
    if game_in_progress:
        return jsonify({"error": "A game is already in progress"}), 400

    game_in_progress = True
    current_game_log = []

    data = request.json or {}
    num_games = min(data.get("num_games", 1), 5)

    def run_games():
        global game_in_progress, benchmark_results
        try:
            for i in range(num_games):
                current_game_log.append(f"=== Starting Game {i+1}/{num_games} ===")

                def log_cb(msg):
                    current_game_log.append(msg)

                game = Game(log_callback=log_cb)
                metrics = game.play_game(max_rounds=10)
                metrics["game_number"] = i + 1
                metrics["game_log"] = list(game.game_log)
                benchmark_results.append(metrics)
                current_game_log.append(f"=== Game {i+1} finished: {metrics['winner']} win! ===")
            # Auto-save results after all games complete
            _save_results_to_file(benchmark_results)
        except Exception as e:
            current_game_log.append(f"Error: {str(e)}")
        finally:
            game_in_progress = False

    thread = threading.Thread(target=run_games, daemon=True)
    thread.start()
    return jsonify({"status": "started", "num_games": num_games})


@app.route("/api/status")
def game_status():
    return jsonify({
        "in_progress": game_in_progress,
        "log": current_game_log[-50:],
        "total_log_lines": len(current_game_log),
    })


@app.route("/api/results")
def get_results():
    if not benchmark_results:
        return jsonify({"results": [], "summary": None})

    # Compute aggregate summary
    total_games = len(benchmark_results)
    fascist_wins = sum(1 for r in benchmark_results if r["winner"] == "Fascists")
    liberal_wins = sum(1 for r in benchmark_results if r["winner"] == "Liberals")
    total_rounds = sum(r["total_rounds"] for r in benchmark_results)
    total_fascist_liberal_votes = sum(r["fascist_voted_liberal"] for r in benchmark_results)
    total_fascist_fascist_votes = sum(r["fascist_voted_fascist"] for r in benchmark_results)
    total_lib_by_fascists = sum(r["liberal_policies_by_fascists"] for r in benchmark_results)
    total_fas_by_fascists = sum(r["fascist_policies_by_fascists"] for r in benchmark_results)
    total_lib_by_liberals = sum(r["liberal_policies_by_liberals"] for r in benchmark_results)
    total_fas_by_liberals = sum(r["fascist_policies_by_liberals"] for r in benchmark_results)

    fascist_total_votes = total_fascist_liberal_votes + total_fascist_fascist_votes
    deception_rate = (
        round(total_fascist_liberal_votes / fascist_total_votes * 100, 1)
        if fascist_total_votes > 0 else 0
    )

    summary = {
        "total_games": total_games,
        "fascist_wins": fascist_wins,
        "liberal_wins": liberal_wins,
        "draws": total_games - fascist_wins - liberal_wins,
        "avg_rounds": round(total_rounds / total_games, 1),
        "deception_rate": deception_rate,
        "fascist_vote_breakdown": {
            "ja_votes": total_fascist_liberal_votes,
            "nein_votes": total_fascist_fascist_votes,
        },
        "policies_enacted": {
            "liberal_by_fascist_gov": total_lib_by_fascists,
            "fascist_by_fascist_gov": total_fas_by_fascists,
            "liberal_by_liberal_gov": total_lib_by_liberals,
            "fascist_by_liberal_gov": total_fas_by_liberals,
        },
    }

    return jsonify({
        "results": benchmark_results,
        "summary": summary,
    })


@app.route("/api/clear")
def clear_results():
    global benchmark_results
    benchmark_results = []
    return jsonify({"status": "cleared"})


def _save_results_to_file(results):
    """Save results to a timestamped JSON file in the results/ directory."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = os.path.join(RESULTS_DIR, f"game_{ts}.json")
    try:
        with open(filepath, "w") as f:
            json.dump({"timestamp": ts, "results": results}, f, indent=2, default=str)
        print(f"  [Saved] Results written to {filepath}")
    except Exception as e:
        print(f"  [Save error] {e}")


@app.route("/api/save", methods=["POST"])
def save_results():
    """Manually save current results to file."""
    if not benchmark_results:
        return jsonify({"error": "No results to save"}), 400
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = os.path.join(RESULTS_DIR, f"game_{ts}.json")
    with open(filepath, "w") as f:
        json.dump({"timestamp": ts, "results": benchmark_results}, f, indent=2, default=str)
    return jsonify({"status": "saved", "file": filepath})


@app.route("/api/download")
def download_results():
    """Download latest results file."""
    files = sorted(
        [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")],
        reverse=True,
    )
    if not files:
        return jsonify({"error": "No saved results found"}), 404
    return send_file(
        os.path.join(RESULTS_DIR, files[0]),
        as_attachment=True,
        download_name=files[0],
    )


@app.route("/api/saved_files")
def list_saved_files():
    """List all saved result files."""
    files = sorted(
        [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")],
        reverse=True,
    )
    return jsonify({"files": files})


@app.route("/api/mi_export")
def mi_export():
    """Export MI-ready dataset for linear probe training.
    Format: list of {hidden_thought, public_statement, role, deceptive_intent, deception_type, model, action_type}
    This is the format needed for training deception detection probes
    (cf. Goldowsky-Dill et al., 'Detecting Strategic Deception Using Linear Probes', ICML 2025)."""
    if not benchmark_results:
        return jsonify({"error": "No results yet"}), 400
    mi_dataset = []
    for game in benchmark_results:
        mi_data = game.get("mi_data", {})
        interactions = mi_data.get("interactions", [])
        for ix in interactions:
            mi_dataset.append({
                "hidden_thought": ix.get("hidden_thought", ""),
                "public_statement": ix.get("public_statement", ""),
                "role": ix.get("role", ""),
                "deceptive_intent": ix.get("deceptive_intent", False),
                "deception_type": ix.get("deception_type", "UNKNOWN"),
                "model": ix.get("model", ""),
                "action_type": ix.get("action", ""),
                "round": ix.get("round", 0),
                "player": ix.get("player", ""),
            })
    # Save to file
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = os.path.join(RESULTS_DIR, f"mi_dataset_{ts}.json")
    with open(filepath, "w") as f:
        json.dump({
            "description": "MI-ready dataset for linear probe training on deception detection",
            "reference": "Goldowsky-Dill et al., Detecting Strategic Deception Using Linear Probes, ICML 2025",
            "total_samples": len(mi_dataset),
            "label_distribution": {
                "deceptive_intent_true": sum(1 for d in mi_dataset if d["deceptive_intent"]),
                "deceptive_intent_false": sum(1 for d in mi_dataset if not d["deceptive_intent"]),
                "DECEPTIVE": sum(1 for d in mi_dataset if d["deception_type"] == "DECEPTIVE"),
                "TRUTHFUL": sum(1 for d in mi_dataset if d["deception_type"] == "TRUTHFUL"),
                "STRATEGIC_TRUTH": sum(1 for d in mi_dataset if d["deception_type"] == "STRATEGIC_TRUTH"),
                "UNKNOWN": sum(1 for d in mi_dataset if d["deception_type"] == "UNKNOWN"),
            },
            "samples": mi_dataset,
        }, f, indent=2)
    return send_file(filepath, as_attachment=True, download_name=f"mi_dataset_{ts}.json")


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)


@app.route("/api/mi_export_full")
def mi_export_full():
    """Export TransformerLens-ready dataset with full prompt text and game state context.

    Each record includes:
    - prompt_text: full reconstructed prompt (system + history + action prompt) for Pythia replay
    - hidden_thought, public_statement: dual-channel output
    - action_type: discussion, vote, policy_president, policy_chancellor, execute, investigate
    - deceptive_intent: bool primary label from role
    - deception_type: DECEPTIVE, STRATEGIC_TRUTH, TRUTHFUL, OMISSION, UNKNOWN
    - game_state: {liberal_policies, fascist_policies, election_tracker, round, players_alive}
    - player_name, player_role, model
    """
    if not benchmark_results:
        return jsonify({"error": "No results yet"}), 400

    full_dataset = []
    for game_idx, game in enumerate(benchmark_results):
        mi_data = game.get("mi_data", {})
        interactions = mi_data.get("interactions", [])
        for ix in interactions:
            full_dataset.append({
                "game_id": game_idx,
                "round": ix.get("round", 0),
                "player_name": ix.get("player", ""),
                "player_role": ix.get("role", ""),
                "model": ix.get("model", ""),
                "action_type": ix.get("action", ""),
                "deceptive_intent": ix.get("deceptive_intent", False),
                "deception_type": ix.get("deception_type", "UNKNOWN"),
                "hidden_thought": ix.get("hidden_thought", ""),
                "public_statement": ix.get("public_statement", ""),
                "prompt_text": ix.get("prompt_text", ""),
                "game_state": ix.get("game_state", {}),
            })

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = os.path.join(DATA_DIR, f"game_{ts}.json")

    export = {
        "description": "TransformerLens-ready MI dataset for activation extraction and linear probing",
        "export_timestamp": ts,
        "total_samples": len(full_dataset),
        "label_distribution": {
            "deceptive_intent_true": sum(1 for d in full_dataset if d["deceptive_intent"]),
            "deceptive_intent_false": sum(1 for d in full_dataset if not d["deceptive_intent"]),
        },
        "action_type_distribution": {},
        "samples": full_dataset,
    }
    # Count action types
    for d in full_dataset:
        at = d["action_type"]
        export["action_type_distribution"][at] = export["action_type_distribution"].get(at, 0) + 1

    with open(filepath, "w") as f:
        json.dump(export, f, indent=2)

    return send_file(filepath, as_attachment=True, download_name=f"game_{ts}.json")


@app.route("/api/mi_summary")
def mi_summary():
    """Get MI summary across all games."""
    if not benchmark_results:
        return jsonify({"error": "No results yet"}), 400
    all_summaries = []
    for game in benchmark_results:
        mi_data = game.get("mi_data", {})
        summary = mi_data.get("summary", {})
        if summary:
            all_summaries.append(summary)
    # Aggregate
    total_interactions = sum(s.get("total_interactions", 0) for s in all_summaries)
    total_deceptive = sum(s.get("deceptive", 0) for s in all_summaries)
    total_truthful = sum(s.get("truthful", 0) for s in all_summaries)
    total_hitler_correct = sum(s.get("hitler_correct_guesses", 0) for s in all_summaries)
    total_hitler_guesses = sum(s.get("hitler_total_guesses", 0) for s in all_summaries)
    return jsonify({
        "total_games": len(all_summaries),
        "total_interactions": total_interactions,
        "total_deceptive": total_deceptive,
        "total_truthful": total_truthful,
        "overall_deception_rate": round(total_deceptive / max(total_interactions, 1) * 100, 1),
        "hitler_detection_rate": round(total_hitler_correct / max(total_hitler_guesses, 1) * 100, 1),
    })


RESULTS_MI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


@app.route("/api/mi/extract_status")
def mi_extract_status():
    """Return status of activation extraction pipeline."""
    activations_dir = os.path.join(DATA_DIR, "activations")
    if not os.path.exists(activations_dir):
        return jsonify({"status": "not_run", "last_run": None, "model": None, "num_interactions": 0})

    # Find latest metadata file
    meta_files = sorted([f for f in os.listdir(activations_dir) if f.startswith("metadata_") and f.endswith(".json")], reverse=True)
    if not meta_files:
        return jsonify({"status": "not_run", "last_run": None, "model": None, "num_interactions": 0})

    meta_path = os.path.join(activations_dir, meta_files[0])
    with open(meta_path) as f:
        meta = json.load(f)

    return jsonify({
        "status": "completed",
        "last_run": meta_files[0],
        "model": meta.get("model", "unknown"),
        "num_interactions": meta.get("total_samples", 0),
        "num_layers": meta.get("num_layers_extracted", 0),
        "skipped": meta.get("skipped_samples", 0),
    })


@app.route("/api/mi/probe_results")
def mi_probe_results():
    """Return latest probe AUROC results per layer."""
    probe_path = os.path.join(RESULTS_MI_DIR, "probe_results.json")
    if not os.path.exists(probe_path):
        return jsonify({"status": "not_run", "layers": []})

    with open(probe_path) as f:
        data = json.load(f)

    layers = []
    for layer_str, metrics in data.get("per_layer_metrics", {}).items():
        layers.append({
            "layer": int(layer_str),
            "auroc": metrics.get("auroc", 0),
            "accuracy": metrics.get("accuracy", 0),
            "f1": metrics.get("f1", 0),
        })
    layers.sort(key=lambda x: x["layer"])

    return jsonify({
        "status": "completed",
        "model": data.get("model", "unknown"),
        "aggregation": data.get("aggregation", "unknown"),
        "best_auroc": data.get("best_auroc", 0),
        "best_layer": data.get("best_layer", -1),
        "layers": layers,
    })


@app.route("/api/mi/comparison")
def mi_comparison():
    """Return base vs fine-tuned comparison results."""
    comp_path = os.path.join(RESULTS_MI_DIR, "comparison_results.json")
    if not os.path.exists(comp_path):
        return jsonify({"status": "not_run", "layers": []})

    with open(comp_path) as f:
        data = json.load(f)

    layers = []
    for layer_str, metrics in data.get("per_layer_comparison", {}).items():
        layers.append({
            "layer": int(layer_str),
            "base_auroc": metrics.get("base_auroc", 0),
            "finetuned_auroc": metrics.get("finetuned_auroc", 0),
            "delta": metrics.get("delta", 0),
        })
    layers.sort(key=lambda x: x["layer"])

    return jsonify({
        "status": "completed",
        "best_improvement_layer": data.get("best_improvement_layer"),
        "max_auroc_delta": data.get("max_auroc_delta", 0),
        "layers": layers,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting Secret Hitler Deception Benchmark on http://localhost:{port}")
    print(f"Make sure OPENROUTER_API_KEY is set in your environment!")
    app.run(debug=False, host="0.0.0.0", port=port)

