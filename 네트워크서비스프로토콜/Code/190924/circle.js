var {center, radius} = require('./cir1');
var area = require('./cir2');
var round = require('./cir3');
var translate = require('./cir4');

console.log('원의 중심 좌표는 (' + center.x + "," + center.y + ')');
console.log("원의 반지름은 " + radius);

// const area = (radius) => (Math.PI * radius * radius);
// const round = (radius) => (Math.PI * 2 * radius);
// const translate = (x,y) => {
//   circle.center.x += x;
//   circle.center.y += y;
// };

console.log("원의 면적은 " + area(radius).toFixed(2) + "입니다");
console.log("원의 둘레는 " + round(radius).toFixed(2) + "입니다");

translate(1,2);
console.log("(1,2) 이동한 원의 중심좌표는 (" + center.x + "," + center.y + ")");
