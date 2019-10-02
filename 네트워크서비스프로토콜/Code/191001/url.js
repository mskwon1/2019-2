const url = require('url');

const { URL } = url;
const myURL = new URL('https://cs.kookmin.ac.kr/major/curriculum2019');
console.log('new URL() : ', myURL);
console.log('url.format() : ', url.format(myURL));
console.log('--------------------------------------');
const parsedurl = url.parse('https://cs.kookmin.ac.kr/major/curriculum2019');
console.log('url.pare() : ', parsedurl);
console.log('url.format() : ', url.format(parsedurl));
