module.exports = {
  //<link rel='stylesheet' type='text/css' href='/css/style.css'>
  HTML:function(schedule_list, body){
    return `
    <!doctype html>
    <html>
    <head>
      <link rel='stylesheet' href='/css/style.css'>
      <title>나의 여행 다이어리</title>
      <meta charset = "utf-8">
    </head>
    <body>
      <div class = "sidenav">
        <div class = "submenu">여행일정공유
          ${schedule_list}
          <a id='new_schedule' href="/create_schedule">새로만들기</a>
        </div>
        <div class= "submenu">추가하기
          <a href="/add_place">여행지 추가하기</a>
          <a href="/add_activity">활동 추가하기</a>
          <a href="/delete_place">여행지 삭제하기</a>
          <a href="/delete_activity">활동 삭제하기</a>
        </div>
      </div>
      <div class = "main">
        <h1 align = 'center' id='main_title'><a href="/">나의 여행 다이어리</a></h1>
        ${body}
      </div>
    </body>
    </html>
    `;
  },schedule_list:function(schedules){
    var list = '';

    var i = 0;
    while(i < schedules.length){
      list = list + `<a href="/schedule/${schedules[i].SCHEDULE_ID}">${schedules[i].SCHEDULE_NAME}</a>`;
      i = i + 1;
    }

    return list;
  },timebox:function(default_time) {
    var html = '';

    var hour, min;
    for (var i=0;i<49;i++) {
      if(i < 20) {
        hour = '0' + parseInt(i/2);
      } else {
        hour = '' + parseInt(i/2);
      }

      if (i % 2 == 0) {
        min = '00'
      } else {
        min = '30'
      }

      time = hour + ':' + min + ':00';

      if (default_time == time) {
        html += `<option value=${time} selected="selected">${time}</option>`
      } else {
        html += `<option value=${time}>${time}</option>`
      }
    }

    return html
  },placeCombobox:function(values) {
    var html = ''
    for (var i=0;i<values.length;i++) {
      html += `<option value=${values[i].PLACE_ID}>${values[i].PLACE_NAME}</option>`

    }

    return html;
  },placeComboboxSub:function(values, default_val) {
    var html = '';
    var current_country = '';
    for (var i=0;i<values.length;i++) {
      if (current_country != values[i].PLACE_COUNTRY) {
        html += `</optgroup>
                  <optgroup label="${values[i].PLACE_COUNTRY}">`
        current_country = values[i].PLACE_COUNTRY
      }
      if (default_val == values[i].PLACE_ID) {
        html += `<option value=${values[i].PLACE_ID} selected="selected">${values[i].PLACE_NAME}</option>`
      } else {
        html += `<option value=${values[i].PLACE_ID}>${values[i].PLACE_NAME}</option>`
      }

    }

    return html;
  },activityCombobox:function(values) {
    var html = ''
    for (var i=0;i<values.length;i++) {
      html += `<option value=${values[i].ACTIVITY_ID}>${values[i].ACTIVITY_NAME}</option>`
    }

    return html;
  },scheduleInfo:function(schedule) {
    return `
      <div class='schedule_name'>[${schedule.SCHEDULE_COUNTRY}] ${schedule.SCHEDULE_NAME}</div>
      <div class='schedule_description'>✈️${schedule.SCHEDULE_DESCRIPTION}</div>
      <hr>
    `
  }
}
