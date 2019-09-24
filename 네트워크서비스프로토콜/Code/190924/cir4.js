var {center, radius}  = require('./cir1');

function translate(x_, y_) {
  center.x += x_;
  center.y += y_;
}

module.exports = translate;
