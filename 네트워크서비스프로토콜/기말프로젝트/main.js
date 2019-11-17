var http = require('http');
var mysql = require('mysql');
var pw = require('./pw.js');
var fs = require('fs');
var qs = require('querystring');
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
        db.query(`SELECT * FROM consists JOIN activity USING (ACTIVITY_ID) JOIN place USING (PLACE_ID) WHERE SCHEDULE_ID = ${queryData.id}
                      ORDER BY CONSISTS_DAY, CONSISTS_TIME`, function(err,consists) {
          let promise = new Promise(function(resolve, reject) {
            body += addBody(body, consists, schedule_name, schedule_description, queryData.id)
            resolve(body)
          })

          function addBody(body, consists, schedule_name, schedule_description, id) {
            body += `<div class='schedule_name'>${schedule_name}</div>
                      <div class='schedule_description'>✈️${schedule_description}</div>
                      <hr>
                      <div class='controls'>
                        <a href='/add_consist?id=${id}'>일정 추가하기</a><br>
                        <a href='/update_schedule?id=${id}'>일정 수정하기</a>
                      </div>
                      `;
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
  } else if (pathname == '/create_schedule') {
      db.query('SELECT * FROM schedule', function(error, schedules) {
        var body = `
        <form action="/create_schedule_process" method="post">
          <p>이름</p>
          <p><input type="text" name="name" placeholder="Schedule Name"></p>
          <p>설명</p>
          <p><textarea name="description" placeholder="Schedule Description"></textarea></p>
          <p><input type="submit" value="저장"></p>
        </form>
        `
        var schedule_list = template.schedule_list(schedules);
        var review_list = template.review_list(schedules);
        var html = template.HTML(schedule_list, review_list, '', body);
        response.writeHead(200, {'Content-Type': 'text/html'});
        response.end(html);
      })
  } else if (pathname == '/create_schedule_process') {
      var body = '';
      request.on('data', function(data) {
        body = body + data;
      })
      request.on('end', function() {
        var post = qs.parse(body);
        var name = post.name;
        var description  = post.description;
        if (name != undefined && description != undefined &&
            name.length <= 45 && description.length <= 90) {
          db.query(`INSERT INTO schedule (SCHEDULE_NAME, SCHEDULE_DESCRIPTION)
                    VALUES(?,?)`, [name, description], function(err, result) {
            if (err) {
              throw err;
            }
            response.writeHead(302, {Location:`/schedule?id=${result.insertId}`})
            response.end()
          })
        } else {
          alert('이름은 45자 이하, 설명은 90자이하여야합니다')
          response.writeHead(302, {Location:'/create_schedule'})
          response.end()
        }
      })
  } else if (pathname == '/update_schedule') {
      var body = '';
      db.query(`SELECT * FROM schedule WHERE SCHEDULE_ID = ${queryData.id}`, function(err2,schedule) {
        var schedule_name = schedule[0].SCHEDULE_NAME
        var schedule_description = schedule[0].SCHEDULE_DESCRIPTION
        db.query(`SELECT * FROM consists JOIN activity USING (ACTIVITY_ID) JOIN place USING (PLACE_ID) WHERE SCHEDULE_ID = ${queryData.id}
                      ORDER BY CONSISTS_DAY, CONSISTS_TIME`, function(err,consists) {
          let promise = new Promise(function(resolve, reject) {
            body += addBody(body, consists, schedule_name, schedule_description, queryData.id)
            resolve(body)
          })

          function addBody(body, consists, schedule_name, schedule_description, id) {
            body += `<div class='schedule_name'>${schedule_name}</div>
                      <div class='schedule_description'>✈️${schedule_description}</div>
                      <hr>
                      `;
            for (var i=0; i<consists.length; i++) {
              var day = consists[i].CONSISTS_DAY;
              var time = consists[i].CONSISTS_TIME;
              var activity_id = consists[i].ACTIVITY_ID;
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
                body += `<div class="activity_image"><img src=${activity_image}></div>`
              } else {
                body += `<div class="activity_image"></div>`
              }

              body += `
                <div class='controls'>
                  <a href='/update_consist?schedule_id=${queryData.id}&activity_id=${activity_id}&day=${day}&time=${time}'>수정하기</a><br>
                  <a href='/delete_consist?schedule_id=${queryData.id}&activity_id=${activity_id}&day=${day}&time=${time}'>삭제하기</a>
                </div>
              </div>
              `
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
  } else if (pathname == '/add_consist') {
      var sel_place_id = queryData.sel_place_id;
      db.query(`SELECT * FROM schedule WHERE SCHEDULE_ID = ${queryData.id}`, function(err, schedules) {
        var body = `<div class='schedule_name'>${schedules[0].SCHEDULE_NAME}</div>
                  <div class='schedule_description'>✈️${schedules[0].SCHEDULE_DESCRIPTION}</div>
                  <hr>`

        db.query('SELECT * FROM place', function(error, places) {
          if (sel_place_id === undefined) {
            sel_place_id = places[0].PLACE_ID
          }
          db.query(`SELECT * FROM activity WHERE PLACE_ID=${sel_place_id}`, function(err_act, activities) {
            body += `
                <script>
                  window.onload = function() {
                    document.getElementById('place_select').value = ${sel_place_id}
                  }
                </script>
                <div class="activity">
                <div class="text_section">
                  <form action="/get_activities" method="post">
                    장소
                    <select id='place_select' name='place_id' onchange="this.form.submit()">
                      ${template.placeCombobox(places)}
                    </select>
                    <input type="hidden" name="schedule_id" value=${queryData.id}>
                  </form>

                  <form action="/add_consist_process" method="post">
                    <div class="activity_name">
                      할일
                      <input type="hidden" name="???" value="???">
                      <select name='activity_id'>
                        ${template.activityCombobox(activities)}
                      </select>
                      <input type="hidden" name="schedule_id" value=${queryData.id}>
                      </div>
                    <div class="activity_time">시간 <input type="number" name="day">일차 🕒
                          <select name="time">${template.timebox("00:00:00")}</select></div>
                </div>
                <div class="submit_button"><input type ="submit" value="추가"></div>
              </form>`

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
  } else if (pathname == '/add_consist_process') {
      request.on('data', function(data) {
        body = body + data;
      })
      request.on('end', function() {
        var post = qs.parse(body);
        var schedule_id = post.schedule_id;
        var activity_id = post.activity_id;
        var day = post.day;
        var time = post.time;

        db.query(`INSERT INTO consists VALUES(?,?,?,?)`, [activity_id, schedule_id, time, day], function(err, result) {
          if (err) {
            throw err;
          }
          response.writeHead(302, {Location:`/schedule?id=${schedule_id}`})
          response.end()
        })
      })
  } else if (pathname == '/update_consist') {
      var body = '';
      var schedule_id = queryData.schedule_id;
      var activity_id = queryData.activity_id;
      var day = queryData.day;
      var time = queryData.time;

      db.query(`SELECT * FROM schedule WHERE SCHEDULE_ID = ${schedule_id}`, function(err2,schedule) {
        var schedule_name = schedule[0].SCHEDULE_NAME
        var schedule_description = schedule[0].SCHEDULE_DESCRIPTION
        db.query(`SELECT * FROM consists JOIN activity USING (ACTIVITY_ID) JOIN place USING (PLACE_ID)
                      WHERE SCHEDULE_ID = ${schedule_id} AND ACTIVITY_ID = ${activity_id} AND CONSISTS_DAY = ${day} AND CONSISTS_TIME = ?`,
                                  [time], function(err,consists) {

          if (err) {
            throw err;
          }
          body += `<div class='schedule_name'>${schedule_name}</div>
                    <div class='schedule_description'>✈️${schedule_description}</div>
                    <hr>
                    `;
          var activity_name = consists[0].ACTIVITY_NAME;
          var activity_description = consists[0].ACTIVITY_DESCRIPTION;
          var activity_image = consists[0].ACTIVITY_IMAGE;
          var place_name = consists[0].PLACE_NAME;

          body += `
            <div class="activity">
            <div class="text_section">
            <form action="/update_consist_process">
              <div class="activity_time">시간 <input type="number" name="day" placeholder=${day}>일차🕒
                    <select name="time">${template.timebox(time)}</select></div> <br>
              <div class="place_name">${place_name}</div><br>
              <div class="activity_name">활동명<br><input type="text" name="activity_name" placeholder=${activity_name}></div> <br>
              <div class="activity_description">활동내용<br><textarea cols="18" rows="5" placeholder=${activity_description}></textarea></div></div>`

          if (activity_image != null) {
            body += `<div class="activity_image"><img src=${activity_image}></div>`
          } else {
            body += `<div class="activity_image"></div>`
          }

          body += '</form></div>'

        db.query('SELECT * FROM schedule', function(error, schedules) {
          var schedule_list = template.schedule_list(schedules);
          var review_list = template.review_list(schedules);
          var html = template.HTML(schedule_list, review_list, '', body);
          response.writeHead(200, {'Content-Type': 'text/html'});
          response.end(html);
        })
      })
    })
  } else if (pathname == '/delete_consist') {
      var schedule_id = queryData.schedule_id;
      var activity_id = queryData.activity_id;
      var day = queryData.day;
      var time = queryData.time;

      db.query('DELETE FROM consists WHERE SCHEDULE_ID = ? AND ACTIVITY_ID = ? AND CONSISTS_DAY = ? AND CONSISTS_TIME = ?',
                  [schedule_id, activity_id, day, time], function(err, result) {
        if (err) {
          throw err;
        }
        response.writeHead(302, {Location : `/update_schedule?id=${schedule_id}`});
        response.end();
      })
  } else if (pathname == '/get_activities') {
      var body = '';

      request.on('data', function(data) {
        body = body + data;
      })
      request.on('end', function() {
        var post = qs.parse(body);

        response.writeHead(302, {Location:`/add_consist?id=${post.schedule_id}&sel_place_id=${post.place_id}`})
        response.end()
      })
  } else if (pathname == '/review') {
    // TODO
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
