# GitHub Repository Setup Commands

Follow these exact steps to create and push the sql-lineage repository to GitHub.

## Prerequisites

1. Make sure you have Git installed: `git --version`
2. Have your GitHub token ready (with repo permissions)
3. Make sure you're in the project directory: `/Users/adhirpotdar/Work/git-repos/sql-lineage`

## Step 1: Initialize Local Git Repository

```bash
cd /Users/adhirpotdar/Work/git-repos/sql-lineage

# Initialize git repository
git init

# Configure user details for this repository
git config user.name "adhir-potdar"
git config user.email "adhir.potdar@isanasystems.com"
```

## Step 2: Add Files to Git

```bash
# Add all files to staging
git add .

# Check what will be committed
git status

# Create initial commit
git commit -m "Initial commit: SQL Lineage Analyzer

- Complete production-ready SQL lineage analysis tool
- Multi-dialect support (Trino, PostgreSQL, MySQL, SQLite)
- Table and column lineage extraction
- CTE and complex query support
- External metadata provider integration
- Comprehensive test suite
- Rich output formatting options
- Clean architecture with separated concerns

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

## Step 3: Create GitHub Repository

**Option A: Using GitHub CLI (if installed)**
```bash
# Install GitHub CLI if not already installed
# brew install gh  # macOS
# Then authenticate: gh auth login

# Create repository on GitHub
gh repo create adhir-potdar/sql-lineage --public --description "Production-quality SQL lineage analysis tool with multi-dialect support and external metadata integration"
```

**Option B: Using GitHub Web Interface**
1. Go to https://github.com/new
2. Repository name: `sql-lineage`
3. Description: `Production-quality SQL lineage analysis tool with multi-dialect support and external metadata integration`
4. Set to Public
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## Step 4: Connect Local Repository to GitHub

```bash
# Add GitHub remote (replace with your actual repo URL)
git remote add origin https://github.com/adhir-potdar/sql-lineage.git

# Verify remote was added
git remote -v
```

## Step 5: Push to GitHub

**Using Personal Access Token:**

```bash
# Set up credential helper (macOS)
git config --global credential.helper osxkeychain

# Push to GitHub (you'll be prompted for username and token)
# Username: adhir-potdar
# Password: [your-github-token]
git push -u origin main
```

**Alternative: Using Token in URL (one-time)**
```bash
# Replace YOUR_TOKEN with your actual GitHub token
git push https://adhir-potdar:YOUR_TOKEN@github.com/adhir-potdar/sql-lineage.git main
```

## Step 6: Verify Upload

```bash
# Check repository status
git status

# Check remote branches
git branch -r

# View commit history
git log --oneline
```

## Step 7: Set Up Branch Protection (Optional)

Go to your GitHub repository settings:
1. Navigate to: Settings → Branches
2. Add branch protection rule for `main`
3. Enable: "Require status checks to pass before merging"
4. Enable: "Require branches to be up to date before merging"

## Expected File Structure on GitHub

After successful push, your repository should contain:

```
sql-lineage/
├── README.md                     # Project documentation
├── LICENSE                       # MIT license
├── .gitignore                    # Python gitignore
├── requirements.txt              # Dependencies
├── install.sh                    # Setup script
├── REFACTORING_SUMMARY.md        # Architecture documentation
├── GITHUB_SETUP_COMMANDS.md      # This file
├── src/                          # Core analyzer code
│   └── analyzer/
├── tests/                        # Pytest test suite
├── examples/                     # Integration examples
├── test_formatter.py             # Output formatting
├── test_quick.py                 # Quick tests
├── test_simple.py                # Simple tests
└── test_samples.py               # Sample tests
```

## Troubleshooting

### If you get authentication errors:
```bash
# Clear credentials and try again
git config --global --unset credential.helper
git push -u origin main
```

### If you get "remote origin already exists":
```bash
# Remove existing remote and add again
git remote remove origin
git remote add origin https://github.com/adhir-potdar/sql-lineage.git
```

### If you get "branch main does not exist":
```bash
# Create and switch to main branch
git checkout -b main
git push -u origin main
```

### If you need to rename default branch from master to main:
```bash
git branch -m master main
git push -u origin main
```

## Post-Upload Steps

1. **Verify repository**: Visit https://github.com/adhir-potdar/sql-lineage
2. **Check README**: Ensure it displays correctly
3. **Test clone**: Try `git clone` from another location
4. **Add topics**: In GitHub, add relevant topics like: `sql`, `lineage`, `data-engineering`, `sqlglot`, `python`
5. **Create release**: Consider creating v1.0.0 release tag

## Security Notes

- **Never commit tokens**: Your GitHub token should never be committed to the repository
- **Use environment variables**: For automated deployments, use GitHub secrets
- **Review .gitignore**: Ensure no sensitive files are tracked

## Next Steps

After successful upload:
1. Set up GitHub Actions for CI/CD (optional)
2. Configure Dependabot for dependency updates
3. Add contributor guidelines
4. Set up issue templates
5. Consider adding GitHub Pages for documentation

---

**Important**: Replace `YOUR_TOKEN` with your actual GitHub personal access token when running the commands.