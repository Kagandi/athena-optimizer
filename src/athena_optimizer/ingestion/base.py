from typing import Protocol

from athena_optimizer.models import QueryRecord


class QueryIngestor(Protocol):
    def fetch(self) -> list[QueryRecord]:
        """Return query records from a source."""
        ...
