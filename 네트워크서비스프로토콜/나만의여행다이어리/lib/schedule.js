const express = require('express')
const path = require('path')
const pw = require('./pw.js')
const mysql = require('mysql')
const template = require('./template.js')

var app = express()
var router = express.Router()

var db = mysql.createConnection({
  host : 'localhost',
  user : 'root',
  password : pw.pw,
  database : 'travel_schedule'
});
db.connect();

router.get('*', function(request, response, next) {
  db.query('SELECT * FROM schedule', function(error, schedules) {
    request.schedule_list = template.schedule_list(schedules);
    next();
  });
});

// 지정한 schedule 열람
router.get('/:schedule_id', function(request, response) {
  var body = '';
  var schedule_id = path.parse(request.params.schedule_id).base;
  db.query(`SELECT * FROM schedule WHERE SCHEDULE_ID = ${schedule_id}`, function(err_sch,schedule) {
    if (err_sch) {
      throw err_sch;
    }
    db.query(`SELECT * FROM consists JOIN activity USING (ACTIVITY_ID) JOIN place USING (PLACE_ID) WHERE SCHEDULE_ID = ${schedule_id}
                  ORDER BY CONSISTS_DAY, CONSISTS_TIME`, function(err_cons,consists) {
      if(err_cons) {
        throw err_cons;
      }

      // 비동기적 수행
      let promise = new Promise(function(resolve, reject) {
        body += addBody(body, consists)
        resolve(body)
      })

      // schedule에 속하는 consists들을 body에 추가
      function addBody(body, consists) {
        body += template.scheduleInfo(schedule[0])
        body += `<div class='controls'>
                    <a href='${schedule_id}/add_consist'>일정 추가하기</a> |
                    <a href='${schedule_id}/update_schedule'>일정 수정하기</a> |
                    <a href='${schedule_id}/delete_schedule'>여행일정 전체 삭제하기</a>
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
            body += `<div class="activity_image"><img src=/${activity_image}></div></div>`
          } else {
            body += `<div class="activity_image"></div></div>`
          }

        }

        return body
      }

      promise.then(function(contents) {
        var html = template.HTML(request.schedule_list, body);
        response.send(html);
      })
    })
  })
})

// schedule 수정 화면
router.get('/:schedule_id/update_schedule', function(request, response) {
  var body = '';
  var schedule_id = path.parse(request.params.schedule_id).base;
  db.query(`SELECT * FROM schedule WHERE SCHEDULE_ID = ${schedule_id}`, function(err_sch,schedule) {
    if (err_sch) {
      throw err_sch;
    }
    db.query(`SELECT * FROM consists JOIN activity USING (ACTIVITY_ID) JOIN place USING (PLACE_ID) WHERE SCHEDULE_ID = ${schedule_id}
                  ORDER BY CONSISTS_DAY, CONSISTS_TIME`, function(err_cons,consists) {
      if (err_cons) {
        throw err_cons;
      }
      let promise = new Promise(function(resolve, reject) {
        body += addBody(body, consists)
        resolve(body)
      })

      function addBody(body, consists) {
        body += template.scheduleInfo(schedule[0]);
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
            body += `<div class="activity_image"><img src=/${activity_image}></div>`
          } else {
            body += `<div class="activity_image"></div>`
          }

          body += `
            <div class='controls'>
              <a href='update_consist?activity_id=${activity_id}&day=${day}&time=${time}'>수정하기</a> |
              <a href='delete_consist?activity_id=${activity_id}&day=${day}&time=${time}'>삭제하기</a>
            </div>
          </div>
          `
        }

        return body
      }

      promise.then(function(contents) {
        var html = template.HTML(request.schedule_list, body);
        response.send(html);
      })
    })
  })
})

// schedule 삭제 요청처리
router.get('/:schedule_id/delete_schedule', function(request, response) {
  var schedule_id = path.parse(request.params.schedule_id).base;

  db.query(`DELETE FROM schedule WHERE SCHEDULE_ID=${schedule_id}`, function(err, result) {
    if (err) {
      throw err;
    }
    response.redirect('/');
  })
})

// schedule에 새로운 세부일정 추가하는 페이지
router.get('/:schedule_id/add_consist', function(request, response) {
  var schedule_id = path.parse(request.params.schedule_id).base;
  var sel_place_id = request.query.sel_place_id;
  db.query(`SELECT * FROM schedule WHERE SCHEDULE_ID = ${schedule_id}`, function(err_sch, schedules) {
    if (err_sch) {
      response.send('<h1>ERROR ... please contact administrator</h1>')
    }
    var body = template.scheduleInfo(schedules[0])

    var schedule_country = schedules[0].SCHEDULE_COUNTRY;

    db.query(`SELECT * FROM place WHERE PLACE_COUNTRY LIKE ?`, [schedule_country], function(err_plc, places) {
      if (err_plc) {
        throw err_plc;
      }

      // 해당 스케쥴의 나라와 일치하는 place가 없는경우
      if (places[0] == undefined) {
        response.redirect(`/schedule/${schedule_id}`);
      }

      // 일치하는 place가 하나라도 있는 경우
      else {
        // place query가 없는 경우(select에서 선택 X)
        if (sel_place_id === undefined) {
          // 기본값으로 첫번째 option 선택
          sel_place_id = places[0].PLACE_ID
        }

        db.query(`SELECT * FROM activity WHERE PLACE_ID=${sel_place_id}`, function(err_act, activities) {
          if (err_act) {
            throw err_act;
          }
          // 첫번째 폼 : select 중 하나 선택시 place값을 갱신하고 refresh
          // 두번째 폼 : consist를 데이터베이스에 추가
          body += `
              <script>
                window.onload = function() {
                  document.getElementById('place_select').value = ${sel_place_id}
                }
              </script>
              <div class="form_wrapper">
              <form action="/processes/get_activities" method="post">
                장소
                <select id='place_select' name='place_id' onchange="this.form.submit()">
                  ${template.placeCombobox(places)}
                </select>
                <input type="hidden" name="schedule_id" value=${schedule_id}>
              </form>

              <form action="/processes/add_consist_process" method="post">
                  할일
                  <input type="hidden" name="???" value="???">
                  <select name='activity_id'>
                    ${template.activityCombobox(activities)}
                  </select>
                  <input type="hidden" name="schedule_id" value=${schedule_id}>
                  <div class="activity_time">시간 <input type="number" name="day" min="1">일차
                  <select name="time">${template.timebox("00:00:00")}</select></div>
              <div class="submit_button"><input type ="submit" value="추가"></div>
            </form></div>`;

          var html = template.HTML(request.schedule_list, body);
          response.send(html);
        })
      }

    })
  })
})

// consist 삭제 처리
router.get('/:schedule_id/delete_consist', function(request, response) {
    var schedule_id = path.parse(request.params.schedule_id).base;
    var activity_id = request.query.activity_id;
    var day = request.query.day;
    var time = request.query.time;

    db.query('DELETE FROM consists WHERE SCHEDULE_ID = ? AND ACTIVITY_ID = ? AND CONSISTS_DAY = ? AND CONSISTS_TIME = ?',
                [schedule_id, activity_id, day, time], function(err, result) {
      if (err) {
        throw err;
      }
      response.redirect(`/schedule/${schedule_id}/update_schedule`);
    })
})

// consist 수정 폼
router.get('/:schedule_id/update_consist', function(request, response) {
  var body = '';
  var schedule_id = path.parse(request.params.schedule_id).base;
  var activity_id = request.query.activity_id;
  var day = request.query.day;
  var time = request.query.time;

  db.query(`SELECT * FROM schedule WHERE SCHEDULE_ID = ${schedule_id}`, function(err_sch,schedule) {
    if (err_sch) {
      throw err_sch;
    }
    db.query(`SELECT * FROM consists JOIN activity USING (ACTIVITY_ID) JOIN place USING (PLACE_ID)
    WHERE SCHEDULE_ID = ${schedule_id} AND ACTIVITY_ID = ${activity_id} AND CONSISTS_DAY = ${day} AND CONSISTS_TIME = ?`,
                              [time], function(err_act,consists) {
      if (err_act) {
        throw err_act;
      }

      body += template.scheduleInfo(schedule[0]);
      var activity_name = consists[0].ACTIVITY_NAME;
      var activity_description = consists[0].ACTIVITY_DESCRIPTION;
      var activity_image = consists[0].ACTIVITY_IMAGE;
      var place_name = consists[0].PLACE_NAME;

      body += `
        <div class="activity">
        <form action="/processes/update_consist_process" method="post" enctype="multipart/form-data">
          <div class="text_section">
          <div class="activity_time">시간 <input type="number" name="day" value=${day}>일차🕒
                <select name="time">${template.timebox(time)}</select></div> <br>
          <div class="place_name">${place_name}</div><br>
          <input type="hidden" name="day_before" value=${day}>
          <input type="hidden" name="time_before" value=${time}>
          <input type="hidden" name="activity_id" value=${activity_id}>
          <input type="hidden" name="schedule_id" value=${schedule_id}>
          <div class="activity_name">활동명<br>
          <input type="text" name="activity_name" value="${activity_name}"></div> <br>
          <div class="activity_description">활동내용<br><textarea name="activity_description">${activity_description}</textarea></div></div>`

      if (activity_image != null) {
        body += `<div class="activity_image"><img src=/${activity_image}>`
      } else {
        body += `<div class="activity_image">`
      }

      body += `<input type="file" name="activity_image" accept=".png, .jpg, .jpeg, .gif"></div>
      <div class="submit_button"><input type ="submit" value="저장"></div></form></div>`

      var html = template.HTML(request.schedule_list, body);
      response.send(html);
    })
  })
})

module.exports = router
