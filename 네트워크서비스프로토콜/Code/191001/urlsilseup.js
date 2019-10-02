const http = require('http');
const url = require('url');
const { URL } = url;


http.createServer((request, response) => {
  const parsedurl = url.parse(request.url)
  const resource = parsedurl.pathname;

  console.log('resource path=/',resource);

  response.writeHead(200, {'Content-Type':'text/plain; charset=utf-8'});
  if (resource == '/') response.end('안녕하세요');
  else if (resource == '/address') response.end('서울특별시 강남구 논현동 111');
  else if (resource == '/phone') response.end('전화번호');
  else if (resource == '/name') response.end('이름');
  else response.end('404 PAGE NOT FOUND');
}).listen(8080, () => {
  console.log('8080번 포트에서 서버 대기중입니다');
})
