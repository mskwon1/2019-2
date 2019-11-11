var http = require('http');
var mysql = require('mysql');
var pw = require('./pw.js');
var fs = require('fs');
var url = require('url');
var path = require('path');
var template = require('./lib/template.js');
var async = require('async');

var db = mysql.createConnection({
  host : 'localhost',
  user : 'root',
  password : pw.pw,
  database : 'travel_schedule'
});
db.connect();

var app = http.createServer(function(request,response) {
  var _url = request.url;
  var queryData = url.parse(_url, true).query;
  var pathname = url.parse(_url, true).pathname;

  if (pathname === '/') {
    if (queryData.id === undefined) {
      db.query('SELECT * FROM schedule', function(error, schedules) {
        var schedule_list = template.schedule_list(schedules);
        var review_list = template.review_list(schedules);
        var html = template.HTML(schedule_list, review_list, '','');
        response.writeHead(200, {'Content-Type': 'text/html'});
        response.end(html);
      })
    }
  } else if (pathname == '/schedule') {
      var body = '';
      db.query(`SELECT * FROM schedule WHERE SCHEDULE_ID = ${queryData.id}`, function(err2,schedule) {
        var schedule_name = schedule[0].SCHEDULE_NAME
        var schedule_description = schedule[0].SCHEDULE_DESCRIPTION
        db.query(`SELECT * FROM consists JOIN activity USING (ACTIVITY_ID) JOIN place USING (PLACE_ID) WHERE SCHEDULE_ID = ${queryData.id}`, function(err,consists) {
          let promise = new Promise(function(resolve, reject) {
            body += addBody(body, consists, schedule_name, schedule_description)
            resolve(body)
          })

          function addBody(body, consists, schedule_name, schedule_description) {
            body += `<div class='schedule_name'>${schedule_name}</div>
                      <div class='schedule_description'>✈️${schedule_description}</div><hr>`;
            for (var i=0; i<consists.length; i++) {
              var day = consists[i].CONSISTS_DAY;
              var time = consists[i].CONSISTS_TIME;
              var activity_name = consists[i].ACTIVITY_NAME;
              var activity_description = consists[i].ACTIVITY_DESCRIPTION;
              var activity_image = consists[i].ACTIVITY_IMAGE;
              var place_name = consists[i].PLACE_NAME;

              body += `
                <div class="activity">
                <div class="text_section">
                <div class="activity_time">${day}일차 🕒 ${time}</div>
                <div class="place_name">${place_name}</div>
                <div class="activity_name">${activity_name}</div>
                <div class="activity_description">${activity_description}</div></div>`

              if (activity_image != null) {
                body += `<div class="activity_image"><img src=${activity_image}></div></div>`
              } else {
                body += `<div class="activity_image"></div></div>`
              }
            }

            return body
          }

          promise.then(function(contents) {
            db.query('SELECT * FROM schedule', function(error, schedules) {
              var schedule_list = template.schedule_list(schedules);
              var review_list = template.review_list(schedules);
              var html = template.HTML(schedule_list, review_list, '', body);
              response.writeHead(200, {'Content-Type': 'text/html'});
              response.end(html);
            })
          })
        })
      })
  } else if (pathname == '/review') {

  } else if (pathname == '/images') {
    fs.readFile(`./images/${queryData.image}`, function(image_error, data){
      if (image_error) {
        throw image_error;
      }
      response.writeHead(200, {'Content-Type' : 'image/jpeg'});
      response.end(data);
    });
  } else if (pathname == '/css') {
    fs.readFile(`./css/${queryData.css}`, function(css_error, data){
      if (css_error) {
        throw css_error;
      }
      response.writeHead(200, {'Content-Type' : 'text/css'});
      response.end(data);
    });
  } else if (pathname == '/fonts') {
    fs.readFile(`./fonts/${queryData.font}`, function(font_error, data) {
      if (font_error) {
        throw font_error;
      }
      response.writeHead(200, {'Content-Type' : 'aplication/font-sfnt'});
      response.end(data);
    })
  } else {
    response.writeHead(404);
    response.end('<h1>404 NOT FOUND</h1>')
  }
})

app.listen(3000);
