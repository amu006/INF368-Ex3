printf "Enter folder name:"
read dd
ls $dd | while read x; do
    printf "%-32s	%8d\n" "$x" $(ls -L "$dd/$x" | wc -l)
done
