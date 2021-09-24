$(function () {
  $("#loader").hide();
  $("#qgen_form").on("submit", function (e) {
    e.preventDefault(); //prevent form from submitting
    $("#loader").show();
    var url = "/projects/qgen";
    $.ajax({
      type: "POST",
      url: url,
      data: {
        context: $("#context").val(),
        answer: $("#answer").val(),
        csrfmiddlewaretoken: $("input[name=csrfmiddlewaretoken]").val(),
        dataType: "json",
      },

      success: function (response) {
        $("#loader").hide();
        var out = "PREDICTED QUESTION: " + response.question;
        // console.log(out);
        $("#output").text(out);
      },

      failure: function () {
        $("#loader").hide();
        console.log("Errored");
      },
    });
  });
});
