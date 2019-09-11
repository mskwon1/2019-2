function diff(start, end) {
  result = 0;
  for (i=start+1;i<end;i++) {
    result += i;
  }
  return result;
}

console.log(diff(1,10));
  
