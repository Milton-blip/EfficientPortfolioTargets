git ls-files | grep -Ev '^(\.venv/|\.idea/|\.venv312/)' > file_list.txt
git ls-files --others --exclude-standard | grep -Ev '^(\.venv/|\.idea/|\.venv312/)' >> file_list.txt
echo "File list written to file_list.txt (excluding .venv, .idea, .venv312)"