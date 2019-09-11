function factorial(number) {
    if (number == 1) {
      return 1;
    } else {
      return number * factorial(number-1);
    }
}

for (i=1;i<=9;i++) {
    console.log(i + "! = " + factorial(i));
}
