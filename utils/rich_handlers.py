from rich.console import Console
from rich.table import Table



class ModelHandler:
    """Rich handler for model operations."""
    
    def __init__(self, console: Console = None):
        self.console = console or Console()
    
    
    def log_model_loading(self, model_path: str, success: bool = True):
        """Log model loading status."""
        if success:
            self.console.print(f"[green]✅ Model loaded successfully from {model_path}[/green]")
        else:
            self.console.print(f"[red]❌ Failed to load model from {model_path}[/red]")
    
    def log_parameters_count(self, total_params: int, trainable_params: int):
        """Display parameter count information."""
        table = Table(title="📊 Model Parameters", show_header=True, header_style="bold blue")
        table.add_column("Parameter Type", style="cyan")
        table.add_column("Count", style="yellow")
        
        table.add_row("Total Parameters", f"{total_params:,}")
        table.add_row("Trainable Parameters", f"{trainable_params:,}")
        table.add_row("Non-trainable Parameters", f"{total_params - trainable_params:,}")
        
        self.console.print(table)



