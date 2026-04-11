"""Testcontainer fixtures with dynamic port allocation."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass

from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import HttpWaitStrategy, LogMessageWaitStrategy
from testcontainers.postgres import PostgresContainer


@dataclass(frozen=True)
class Container:
    """Connection info for a running testcontainer.

    Attributes:
        host: Container host (always localhost)
        port: Dynamic mapped port
        url: Pre-built connection URL
    """

    host: str
    port: int
    url: str


@contextmanager
def _start_container(
    image: str,
    port: int,
    wait_for_log: str,
) -> Generator[Container]:
    """Start a generic Docker container with dynamic port. Internal building block.

    Args:
        image: Docker image
        port: Internal container port to expose
        wait_for_log: Log message indicating readiness

    Yields:
        Container with http:// URL and dynamic port
    """
    c = (
        DockerContainer(image)
        .with_exposed_ports(port)
        .waiting_for(LogMessageWaitStrategy(wait_for_log))
    )

    with c:
        mapped = int(c.get_exposed_port(port))
        yield Container(
            host="localhost",
            port=mapped,
            url=f"http://localhost:{mapped}",
        )


@contextmanager
def start_postgres(
    password: str,
    image: str = "postgres:15",
    username: str = "postgres",
    dbname: str = "postgres",
) -> Generator[Container]:
    """Start a Postgres container with dynamic port.

    Args:
        password: DB password
        image: Docker image (postgres:15, timescale/timescaledb:latest-pg15)
        username: Database username
        dbname: Database name

    Yields:
        Container with postgresql:// URL and dynamic port
    """
    with PostgresContainer(
        image, username=username, password=password, dbname=dbname
    ) as c:
        port = int(c.get_exposed_port(5432))
        yield Container(
            host="localhost",
            port=port,
            url=f"postgres://{username}:{password}@localhost:{port}/{dbname}",
        )


@contextmanager
def start_mlflow() -> Generator[Container]:
    """Start an MLflow tracking server with dynamic port.

    Yields:
        Container with http:// URL pointing to MLflow UI/API
    """
    c = (
        DockerContainer("ghcr.io/mlflow/mlflow:latest")
        .with_exposed_ports(5000)
        .with_command(
            "mlflow server --host 0.0.0.0 --port 5000"
            " --backend-store-uri sqlite:///mlflow.db"
            " --default-artifact-root file:///tmp/mlflow"
        )
        .waiting_for(LogMessageWaitStrategy("Application startup complete"))
    )

    with c:
        mapped = int(c.get_exposed_port(5000))
        yield Container(
            host="localhost",
            port=mapped,
            url=f"http://localhost:{mapped}",
        )


@contextmanager
def start_pushgateway() -> Generator[Container]:
    """Start a Prometheus Pushgateway with dynamic port.

    Yields:
        Container with http:// URL pointing to Pushgateway
    """
    c = (
        DockerContainer("prom/pushgateway:latest")
        .with_exposed_ports(9091)
        .waiting_for(HttpWaitStrategy(9091, "/metrics").for_status_code(200))
    )

    with c:
        mapped = int(c.get_exposed_port(9091))
        yield Container(
            host="localhost",
            port=mapped,
            url=f"http://localhost:{mapped}",
        )
