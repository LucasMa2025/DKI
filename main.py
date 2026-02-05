"""
DKI Main Entry Point
Provides command-line interface for DKI system
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="DKI - Dynamic KV Injection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py web              Start web UI server
  python main.py api              Start API server only
  python main.py generate-data    Generate experiment data
  python main.py experiment       Run experiments
  python main.py demo             Run interactive demo
        """
    )
    
    parser.add_argument(
        "command",
        choices=["web", "api", "generate-data", "experiment", "demo", "test"],
        help="Command to run"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for web/api server (default: 8080)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for web/api server (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--engine",
        type=str,
        choices=["vllm", "llama", "deepseek", "glm"],
        default=None,
        help="Model engine to use"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Set config path if provided
    if args.config:
        import os
        os.environ["DKI_CONFIG_PATH"] = args.config
    
    if args.command == "web":
        run_web(args)
    elif args.command == "api":
        run_api(args)
    elif args.command == "generate-data":
        run_generate_data()
    elif args.command == "experiment":
        run_experiment()
    elif args.command == "demo":
        run_demo(args)
    elif args.command == "test":
        run_tests()


def run_web(args):
    """Start web UI server."""
    print("Starting DKI Web UI...")
    print(f"Server will be available at http://{args.host}:{args.port}")
    
    import uvicorn
    from dki.web.app import create_app
    
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


def run_api(args):
    """Start API server."""
    print("Starting DKI API Server...")
    print(f"API will be available at http://{args.host}:{args.port}")
    
    import uvicorn
    
    uvicorn.run(
        "dki.web.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=False,
    )


def run_generate_data():
    """Generate experiment data."""
    print("Generating experiment data...")
    
    from dki.experiment.data_generator import ExperimentDataGenerator
    
    generator = ExperimentDataGenerator("./data")
    generator.generate_all()
    generator.generate_alpha_sensitivity_data()
    
    print("Data generation complete!")
    print("Files created in ./data/")


def run_experiment():
    """Run experiments."""
    print("Running experiments...")
    print("Note: This requires a loaded model and may take significant time.")
    
    from dki.experiment.runner import ExperimentRunner, ExperimentConfig
    
    runner = ExperimentRunner()
    config = ExperimentConfig(
        name="CLI Experiment",
        modes=["dki", "rag", "baseline"],
        datasets=["memory_qa"],
        max_samples=10,
    )
    
    try:
        results = runner.run_experiment(config)
        print("\nExperiment Results:")
        print(f"Experiment ID: {results['experiment_id']}")
        print("\nAggregated Metrics:")
        for mode, metrics in results.get('aggregated_metrics', {}).items():
            print(f"  {mode}:")
            for key, value in metrics.items():
                print(f"    {key}: {value}")
    except Exception as e:
        print(f"Experiment failed: {e}")
        print("Make sure you have generated data first: python main.py generate-data")


def run_demo(args):
    """Run interactive demo."""
    print("=" * 50)
    print("DKI Interactive Demo")
    print("=" * 50)
    print()
    
    # Check if we can import torch
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch not installed. Please run: pip install -r requirements.txt")
        return
    
    print()
    print("This demo shows DKI vs RAG comparison.")
    print("Due to model loading requirements, this is a simplified version.")
    print()
    
    # Demo without loading actual models
    from dki.core.memory_router import MemoryRouter
    from dki.core.embedding_service import EmbeddingService
    
    print("Initializing embedding service...")
    embedding_service = EmbeddingService(device="cpu")
    router = MemoryRouter(embedding_service)
    
    print("\nAdding sample memories...")
    memories = [
        ("mem1", "User prefers vegetarian food and is allergic to seafood"),
        ("mem2", "User lives in Beijing and works as a software engineer"),
        ("mem3", "User enjoys hiking and photography on weekends"),
        ("mem4", "User's birthday is on March 15th"),
        ("mem5", "User prefers coffee over tea"),
    ]
    
    for mem_id, content in memories:
        router.add_memory(mem_id, content)
        print(f"  Added: {content[:50]}...")
    
    print("\n" + "=" * 50)
    print("Memory Search Demo")
    print("=" * 50)
    
    queries = [
        "What should I eat for dinner?",
        "What activities can I do this weekend?",
        "When is my birthday?",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = router.search(query, top_k=3)
        print("Retrieved memories:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result.score:.3f}] {result.content[:60]}...")
    
    print("\n" + "=" * 50)
    print("Demo complete!")
    print("For full functionality, start the web UI: python main.py web")
    print("=" * 50)


def run_tests():
    """Run tests."""
    print("Running tests...")
    import pytest
    pytest.main(["tests/", "-v"])


if __name__ == "__main__":
    main()
