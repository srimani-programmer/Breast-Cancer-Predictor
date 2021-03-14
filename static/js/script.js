window.addEventListener("load",function animatedformdown(event){
    const arrows=document.querySelectorAll(".fa-arrow-circle-down")
    arrows.forEach(arrow=>{
        
        arrow.addEventListener('click',function(e)
        {
            const input=arrow.previousElementSibling;
            var value1 = input.value;
            const parent=arrow.parentElement;
            const nextForm=parent.nextElementSibling;
            if(value1==''){alert('please enter input')}
            else{nextslide(parent,nextForm);}
        });


    });
});

window.addEventListener("load",function animatedformup(event){
        const arrows=document.querySelectorAll(".fa-arrow-circle-up")
        arrows.forEach(arrow=>{
            arrow.addEventListener('click',function(e)

            {
                console.log('hi');
                const input=arrow.previousElementSibling;
                const parent=arrow.parentElement;
                const nextForm=parent.previousElementSibling;
                nextslide(parent,nextForm);
            });
    
        });
    
    });

    function nextslide(parent,nextForm){

    parent.classList.add('inactive');
    parent.classList.remove('active');
    nextForm.classList.add('active');
};

