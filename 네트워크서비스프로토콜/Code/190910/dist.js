function dist(p1, p2) {
  x = (p2.x-p1.x);
  y = (p2.y-p1.y);

  return Math.sqrt(x*x + y*y);
}

var p1 = {x:1, y:1};
var p2 = {x:4, y:5};
console.log("(1,1)에서 (4,5)사이의 거리는 " + dist(p1,p2)+ "입니다.");
