#!/bin/bash

# Variables - Replace these with your desired values
#REPO_NAME="QuantumPortfolioPIMC"
COMMIT_MESSAGE="📚 Add performance optimization docs and update README with clean structure"

# Initialize Git repository, set main branch, add all files, commit, create GitHub repo, and push
# Using && ensures each command only runs if the previous one succeeds
# git init && \
# git branch -M main && \
git add . && \
git commit -m "$COMMIT_MESSAGE" && \
# gh repo create "$REPO_NAME" --public --source=. --push && \
git push -u origin main

echo "Successfully initialized Git repository, committed all files, created GitHub repository '$REPO_NAME', and pushed to main branch."