const template = require('./lib/template.js');
const schedule = require('./lib/schedule.js');
const processes = require('./lib/processes.js');
const forms = require('./lib/forms.js');
const pw = require('./lib/pw.js');
const express = require('express');
const mysql = require('mysql');
const bodyParser = require('body-parser');
const compression = require('compression');

var db = mysql.createConnection({
  host : 'localhost',
  user : 'root',
  password : pw.pw,
  database : 'travel_schedule'
});
db.connect();

const app  = express()

// 미들웨어 스택
app.use(express.static('public'));
app.use(bodyParser.urlencoded({extended: false}));
app.use(compression());
app.use('/schedule', schedule)
app.use('/processes', processes)
app.use('/forms', forms)

// request 객체에 schedule_list field 추가(모든 페이지 공통)
app.get('*', function(request, response, next) {
  db.query('SELECT * FROM schedule', function(error, schedules) {
    request.schedule_list = template.schedule_list(schedules);
    next();
  });
});

// 첫 페이지
app.get('/', function(request, response) {
  var html = template.HTML(request.schedule_list,'<div class="imgwrapper"><img src="/images/main.png"><br>떠나자!!!!</div>');
  response.send(html);
})

// 404 ERROR
app.use(function(req, res, next) {
  res.status(404).send('<h1> 404 ERROR : PAGE NOT FOUND </h1>');
});

// 서버 실행
app.listen(3000, () => console.log('Server Running on port number 3000'))
