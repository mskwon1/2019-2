const express = require('express')
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

// 새로운 schedule 생성 폼
router.get('/create_schedule', function(request, response) {
  var body = `
  <form action="/processes/create_schedule_process" method="post">
    <p>이름</p>
    <p><input type="text" name="name" placeholder="Schedule Name" maxlength="100"></p>
    <p>설명</p>
    <p><textarea name="description" placeholder="Schedule Description" maxlength="500"></textarea></p>
    <p>나라이름</p>
    <p><input type="text" name="country" placeholder="Schedule Country" maxlength="45"></p>
    <p><input type="submit" value="저장"></p>
  </form>
  `;

  var html = template.HTML(request.schedule_list, body);
  response.send(html);
})

// place 추가 폼
router.get('/add_place', function(request, response) {
  var body = `
  <form action="/processes/add_place_process" method="post">
    <p>나라이름</p>
    <p><input type="text" name="place_country" placeholder="Place Country" maxlength="45"></p>
    <p>여행지 이름</p>
    <p><input type="text" name="place_name" placeholder="Place Name" maxlength="45"></p>
    <p><input type="submit" value="추가"></p>
  </form>
  `;
  var html = template.HTML(request.schedule_list, body);
  response.send(html);
})

// activity 추가 폼
router.get('/add_activity', function(request, response) {
  db.query('SELECT * FROM place ORDER BY PLACE_COUNTRY', function(err_plc, places) {
    if (err_plc) {
      throw err_plc;
    }
    var body = `
    <form action="/processes/add_activity_process" method="post" enctype="multipart/form-data">
      <p>활동 여행지</p>
      <p><select name="activity_place">${template.placeComboboxSub(places)}</select>
      <p>활동 이름</p>
      <p><input type="text" name="activity_name" placeholder="Activity Name" maxlength="100"></p>
      <p>활동 설명</p>
      <p><textarea name="activity_description" placeholder="Activity Description" maxlength="500"></textarea></p>
      <p>활동 사진</p>
      <input type="file" name="activity_image" accept=".png, .jpg, .jpeg, .gif">
      <p><input type="submit" value="추가"></p>
    </form>
    `;

    var html = template.HTML(request.schedule_list, body);
    response.send(html);
  })
})

// place 삭제 폼
router.get('/delete_place', function(request, response) {
  db.query('SELECT * FROM place', function(err_plc, places) {
    if (err_plc) {
      throw err_plc;
    }

    var body = `
    <form action="/processes/delete_place_process" method="post">
      <p>여행지 이름</p>
      <select name="place_id">${template.placeComboboxSub(places)}</select>
      <p>
      <div class="warning">주의 : 해당 여행지에서 진행되는 활동 데이터가 모두 지워집니다!</div>
      <div class="submit_button"> <input type="submit" value="삭제"> </div>
      </p>
    </form>
    `;
    var html = template.HTML(request.schedule_list, body);
    response.send(html);
  })
})

// activity 삭제 폼
router.get('/delete_activity', function(request, response) {
  var body = ''
  var sel_place_id = request.query.sel_place_id;
  db.query(`SELECT * FROM place ORDER BY PLACE_COUNTRY`, function(err_plc, places) {
    if (err_plc) {
      throw err_plc;
    }

    if (sel_place_id === undefined) {
      sel_place_id = places[0].PLACE_ID
    }
    db.query('SELECT * FROM activity WHERE PLACE_ID = ?', [sel_place_id], function(err_act, activities) {
      if (err_act) {
        throw err_act;
      }

      // 첫번째 폼 : select 중 하나 선택시 place값을 갱신하고 refresh
      // 두번째 폼 : 삭제진행
      body += `
        <form action="/processes/get_activities_place" method="post">
          <p>장소</p>
          <select id='place_select' name='place_id' onchange="this.form.submit()">
            ${template.placeComboboxSub(places, sel_place_id)}
          </select>
        </form>

        <form action="/processes/delete_activity_process" method="post">
            <p>활동명</p>
            <select name='activity_id'>
              ${template.activityCombobox(activities)}
            </select>
            <p>
            <div class="warning">주의 : 해당 활동이 모든 여행일정에서 삭제됩니다!</div>
            <div class="submit_button"><input type ="submit" value="삭제">
            </p>
        </form>
        `;
      var html = template.HTML(request.schedule_list, body);
      response.send(html);
    })
  })
})

module.exports = router
