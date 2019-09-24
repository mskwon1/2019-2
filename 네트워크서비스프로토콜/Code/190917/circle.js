var circle = {
  center : {x:1.0, y:2.0},
  radius : 2.5,
}

console.log('원의 중심 좌표는 (' + circle.center.x + "," + circle.center.y + ')');
console.log("원의 반지름은 " + circle.radius);

const area = (radius) => (Math.PI * radius * radius);
const round = (radius) => (Math.PI * 2 * radius);
const translate = (x,y) => {
  circle.center.x += x;
  circle.center.y += y;
};

console.log("원의 면적은 " + area(circle.radius).toFixed(2) + "입니다");
console.log("원의 둘레는 " + round(circle.radius).toFixed(2) + "입니다");

translate(1,2);
console.log("(1,2) 이동한 원의 중심좌표는 (" + circle.center.x + "," + circle.center.y + ")");
