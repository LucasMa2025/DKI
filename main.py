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
  python main.py web                         Start web UI server
  python main.py api                         Start API server only
  python main.py demo                        Start Demo App (FastAPI web app)
  python main.py demo --port 8080            Demo App on port 8080
  python main.py demo --config config/demo.yaml  Demo App with custom config
  python main.py generate-data               Generate experiment data
  python main.py experiment                  Run experiments
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
    """Start the DKI Demo App (FastAPI web application).

    Launches demo/app.py via uvicorn — this is the full-featured demo
    application with chat, session management, preference editing, and
    DKI Plugin integration.

    Accessible at: http://{host}:{port}
    API docs:       http://{host}:{port}/docs
    Health check:   http://{host}:{port}/api/health
    """
    import os
    import uvicorn

    print("=" * 60)
    print("DKI Demo App")
    print("=" * 60)
    print(f"Host   : {args.host}")
    print(f"Port   : {args.port}")
    print(f"Config : {os.environ.get('DKI_CONFIG_PATH', '(default)')}")
    print()
    print(f"Demo App will be available at http://{args.host}:{args.port}")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    print("=" * 60)
    print()

    # create_demo_app() reads DKI_CONFIG_PATH from env (already set by
    # main() if --config was passed, or exported by start_dki_with_model.sh)
    from demo.app import create_demo_app

    config_path = os.environ.get("DKI_CONFIG_PATH") or None
    app = create_demo_app(config_path=config_path)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


def run_tests():
    """Run tests."""
    print("Running tests...")
    import pytest
    pytest.main(["tests/", "-v"])


if __name__ == "__main__":
    main()
