from sys import stdout as std

from loguru import logger

# 配置logger
logger.remove()

logger.level("DEBUG", color="<cyan><bold>", icon="🐸")
logger.level("INFO", color="<green><bold>", icon="✅")
logger.level("WARNING", color="<yellow><bold>", icon="⚠️")
logger.level("ERROR", color="<red><bold>", icon="❌")
logger.level("CRITICAL", icon="🔥")
logger.level("SUCCESS", color="<fg #00FF00><bold>", icon="🎉")

# TODO: ICON is not shown in console
logger.add(
    std,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> {extra['type]} | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
    level="DEBUG",
    filter=lambda record: record["extra"].get("type") == "app",
    # enqueue=True,
)
logger.add(
    "./logs/app_dev.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} {extra['type']} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="1 day",
    retention="10 days",
    level="DEBUG",
    # enqueue=True,
)
app_logger = logger.bind(type="app")

logger.add(
    "./ml/logs/ml_dev.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} {extra['type']} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="1 day",
    retention="10 days",
    level="DEBUG",
    filter=lambda record: record["extra"].get("type") == "ml"
    # enqueue=True,
)
ml_logger = logger.bind(type="ml")