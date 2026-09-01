import reflex as rx
from reflex.plugins.sitemap import SitemapPlugin


config = rx.Config(
    app_name="orderflow_reflex",
    frontend_port=8080,
    backend_port=8000,
    env_file=".env",
    state_auto_setters=True,
    disable_plugins=[SitemapPlugin],
)
