#!/bin/bash

# Variables - Replace these with your desired values
#REPO_NAME="QuantumPortfolioPIMC"
COMMIT_MESSAGE="⚙️ Integrate configuration system into comprehensive test framework

- Updated SingleStrategyComprehensiveTest to use config_manager profiles
- Added command-line profile selection (--profile argument)  
- Replaced hard-coded bootstrap/validation configs with dynamic loading
- Bootstrap now uses profile-specific n_sims, method, and batch_size
- Monte Carlo validation uses profile-specific sample counts
- Added proper conversion from BootstrapProfile to BootstrapConfig
- Enhanced error handling for profile loading with fallback defaults
- Comprehensive test now displays active profile and settings
- Performance timing shows optimization effects of different profiles
- All configuration integration tested and working correctly"

# Initialize Git repository, set main branch, add all files, commit, create GitHub repo, and push
# Using && ensures each command only runs if the previous one succeeds
# git init && \
# git branch -M main && \

# gh repo create "$REPO_NAME" --public --source=. --push && \


git add . && \
git commit -m "$COMMIT_MESSAGE" && \

git push -u origin main

echo "Successfully initialized Git repository, committed all files, created GitHub repository '$REPO_NAME', and pushed to main branch."