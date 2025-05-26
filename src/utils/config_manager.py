"""
Configuration Management System

Centralized configuration loading for bootstrap profiles, strategy parameters,
and batch testing configurations.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class BootstrapProfile:
    """Bootstrap configuration profile."""
    n_sims: int
    batch_size: int
    block_length: int
    method: str
    confidence_level: float
    description: str

@dataclass
class ValidationProfile:
    """Monte Carlo validation configuration profile."""
    n_bootstrap_samples: int
    n_permutation_samples: int
    confidence_level: float
    alpha: float
    description: str

class ConfigManager:
    """Centralized configuration manager."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        if config_dir is None:
            # Default to config directory in project root
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config"
        
        self.config_dir = Path(config_dir)
        self._bootstrap_configs = None
        self._validation_configs = None
        
    def load_bootstrap_config(self, profile: str = "development") -> BootstrapProfile:
        """Load bootstrap configuration for specified profile."""
        if self._bootstrap_configs is None:
            self._load_bootstrap_configs()
            
        if profile not in self._bootstrap_configs:
            available = list(self._bootstrap_configs.keys())
            raise ValueError(f"Profile '{profile}' not found. Available: {available}")
            
        config = self._bootstrap_configs[profile]
        return BootstrapProfile(**config)
    
    def load_validation_config(self, profile: str = "development") -> ValidationProfile:
        """Load validation configuration for specified profile."""
        if self._validation_configs is None:
            self._load_validation_configs()
            
        if profile not in self._validation_configs:
            # Create default validation profiles based on bootstrap profiles
            bootstrap_profile = self.load_bootstrap_config(profile)
            
            # Scale validation samples based on bootstrap configuration
            scale_factor = bootstrap_profile.n_sims / 100  # Base on 100 sims
            
            return ValidationProfile(
                n_bootstrap_samples=max(50, int(bootstrap_profile.n_sims * 0.2)),
                n_permutation_samples=max(50, int(bootstrap_profile.n_sims * 0.2)),
                confidence_level=bootstrap_profile.confidence_level,
                alpha=1 - bootstrap_profile.confidence_level,
                description=f"Auto-generated from {profile} bootstrap profile"
            )
            
        config = self._validation_configs[profile]
        return ValidationProfile(**config)
    
    def _load_bootstrap_configs(self):
        """Load bootstrap configurations from YAML file."""
        config_file = self.config_dir / "bootstrap_configs.yaml"
        
        if not config_file.exists():
            # Create default configurations
            self._create_default_bootstrap_configs()
            
        with open(config_file, 'r') as f:
            self._bootstrap_configs = yaml.safe_load(f)
    
    def _load_validation_configs(self):
        """Load validation configurations."""
        config_file = self.config_dir / "validation_configs.yaml"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                self._validation_configs = yaml.safe_load(f)
        else:
            self._validation_configs = {}
    
    def _create_default_bootstrap_configs(self):
        """Create default bootstrap configurations if none exist."""
        default_configs = {
            'development': {
                'n_sims': 100,
                'batch_size': 50,
                'block_length': 10,
                'method': 'IID',
                'confidence_level': 0.95,
                'description': 'Optimized for fast development iteration'
            },
            'production': {
                'n_sims': 1000,
                'batch_size': 100,
                'block_length': 10,
                'method': 'BLOCK',
                'confidence_level': 0.95,
                'description': 'Full statistical rigor for production analysis'
            },
            'quick_test': {
                'n_sims': 50,
                'batch_size': 25,
                'block_length': 5,
                'method': 'IID',
                'confidence_level': 0.90,
                'description': 'Minimal configuration for quick validation'
            },
            'research': {
                'n_sims': 2000,
                'batch_size': 200,
                'block_length': 20,
                'method': 'STATIONARY',
                'confidence_level': 0.99,
                'description': 'Extended analysis for research purposes'
            }
        }
        
        config_file = self.config_dir / "bootstrap_configs.yaml"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            yaml.dump(default_configs, f, default_flow_style=False, sort_keys=False)
    
    def list_profiles(self) -> Dict[str, Dict[str, Any]]:
        """List all available configuration profiles."""
        if self._bootstrap_configs is None:
            self._load_bootstrap_configs()
            
        return {
            'bootstrap': list(self._bootstrap_configs.keys()),
            'descriptions': {
                profile: config['description'] 
                for profile, config in self._bootstrap_configs.items()
            }
        }
    
    def get_performance_summary(self) -> Dict[str, str]:
        """Get performance summary for each profile."""
        profiles = self.list_profiles()
        
        performance_map = {
            'quick_test': '< 1 second per symbol',
            'development': '2-5 seconds per symbol', 
            'production': '20-50 seconds per symbol',
            'research': '1-5 minutes per symbol'
        }
        
        return {
            profile: performance_map.get(profile, 'Unknown')
            for profile in profiles['bootstrap']
        }

# Global configuration manager instance
config_manager = ConfigManager()

# Convenience functions
def get_bootstrap_config(profile: str = "development") -> BootstrapProfile:
    """Get bootstrap configuration for specified profile."""
    return config_manager.load_bootstrap_config(profile)

def get_validation_config(profile: str = "development") -> ValidationProfile:
    """Get validation configuration for specified profile."""
    return config_manager.load_validation_config(profile)

def list_available_profiles() -> Dict[str, Any]:
    """List all available configuration profiles."""
    return config_manager.list_profiles()

def get_performance_summary() -> Dict[str, str]:
    """Get performance summary for all profiles."""
    return config_manager.get_performance_summary()
