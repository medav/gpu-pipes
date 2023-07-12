
# set -x
NI=100000000
NW=4

test () {
    ./atomics.elf $1 $NW $2 $NI
}

# ./atomics.elf $(($NV * $C)) $NW $NV $NI

# test NB NV


echo "cont_1 = {"
test 1 1
test 2 2
test 4 4
test 8 8
test 16 16
test 32 32
test 64 64
test 108 108
echo "}"

echo "cont_2 = {"
test 2 1
test 4 2
test 8 4
test 16 8
test 32 16
test 64 32
test 108 54
test 216 108
echo "}"

echo "cont_4 = {"
test 4 1
test 8 2
test 16 4
test 32 8
test 64 16
test 108 32
test 216 64
test 432 108
echo "}"

echo "cont_8 = {"
test 8 1
test 16 2
test 32 4
test 64 8
test 108 16
test 216 32
test 432 64
test 864 108
echo "}"

echo "cont_16 = {"
test 16 1
test 32 2
test 64 4
test 108 8
test 216 16
test 432 32
test 864 64
echo "}"

echo "cont_32 = {"
test 32 1
test 64 2
test 108 3
echo "}"

echo "cont_64 = {"
test 64 1
test 108 2
echo "}"

echo "cont_108 = {"
test 108 1
echo "}"


