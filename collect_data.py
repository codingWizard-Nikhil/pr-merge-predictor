from github import Github
import pandas as pd
from config import GITHUB_TOKEN


REPOS = [
    "microsoft/vscode",
    "pytorch/pytorch"
]

PRS_PER_REPO = 5000


def collect_pr_data(repo_name, g, max_prs):
    """Fetches PR data from a repo"""

    print(f"\nFetching PRs from {repo_name}...")

    repo = g.get_repo(repo_name)

    pulls = repo.get_pulls(state='all', sort='created', direction='desc')

    pr_data = []
    count = 0

    for pr in pulls:
        if count >= max_prs:
            break
        
        data = {

        'repo': repo_name,
        'pr_number': pr.number,
        'state': pr.state,
        'merged': 1 if pr.merged else 0,
        'created_at': pr.created_at,
        'closed_at': pr.closed_at,
        'additions': pr.additions,
        'deletions': pr.deletions,
        'changed_files': pr.changed_files,
        'commits': pr.commits,
        'comments': pr.comments,
        'title': pr.title,
        'body': pr.body if pr.body else "",
        'user_login': pr.user.login if pr.user else None

        }

        pr_data.append(data)
        count += 1

        if count % 100 == 0:
            print(f"Collected {count} PRs so far")
    
    print(f"Completed: {count} PRs from {repo_name}")
    return pr_data



if __name__ == "__main__":
    print("Starting GitHub PR data collection...")
    print(f"Repos: {REPOS}")
    print(f"Target: {PRS_PER_REPO} PRs per repo")

    g = Github(GITHUB_TOKEN)


    all_data = []

    for repo_name in REPOS:
        dataset = collect_pr_data(repo_name, g, PRS_PER_REPO)
        all_data.extend(dataset)
        
    df = pd.DataFrame(all_data)
    df.to_csv('github_prs_raw.csv', index=False)


    print(f"\n✓ Done! Collected {len(all_data)} PRs total")
    print(f"✓ Saved to github_prs_raw.csv")