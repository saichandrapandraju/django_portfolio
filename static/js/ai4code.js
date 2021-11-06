$(function () {
  $("#loader").hide();
  $("#time").hide();
  $("#ai4code_form").on("submit", function (e) {
    e.preventDefault(); //prevent form from submitting
    $("#loader").show();
    $("#time").show();
    $("#output").text("");
    var url = "/projects/ai4code";
    $.ajax({
      type: "POST",
      url: url,
      data: {
        code: $("#code").val(),
        language: $("#language").val(),
        csrfmiddlewaretoken: $("input[name=csrfmiddlewaretoken]").val(),
        dataType: "json",
      },

      success: function (response) {
        $("#loader").hide();
        $("#time").hide();
        var out = response.output;
        // console.log(out);
        $("#output").text(out);
      },

      failure: function () {
        $("#loader").hide();
        $("#time").hide();
        $("#output").val("<ERROR>Something Went Wrong.<ERROR>");
        console.log("Errored");
      },
    });
  });
});
