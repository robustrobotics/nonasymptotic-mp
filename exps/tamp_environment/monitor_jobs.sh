for file in *.out; do echo "$file: $(tail -n 1 "$file")"; done