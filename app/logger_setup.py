import logging
import os
from dotenv import load_dotenv

from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

load_dotenv()


def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("churn-api")

    provider = LoggerProvider(resource=Resource.create({"service.name": "churn-api"}))
    set_logger_provider(provider)

    exporter = OTLPLogExporter(
        endpoint="https://in-otel.hyperdx.io/v1/logs",
        headers={"authorization": os.getenv("HYPERDX_API_KEY")},
    )

    provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    handler = LoggingHandler(logger_provider=provider)
    logger.addHandler(handler)

    return logger
