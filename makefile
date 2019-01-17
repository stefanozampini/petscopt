# -*- mode: makefile-gmake -*-

override PETSCOPT_DIR := $(CURDIR)
include ./lib/petscopt/conf/variables
-include makefile.in

all-parallel: ruler := $(subst -,==========,--------)
all-parallel:
	-@echo $(ruler)
	-@echo "Building PETSCOPT (GNU Make - $(MAKE_NP) build jobs)"
	-@echo "Using PETSCOPT_DIR=$(PETSCOPT_DIR)"
	-@echo "Using PETSCOPT_ARCH=$(PETSCOPT_ARCH)"
	-@echo "Using PETSC_DIR=$(PETSC_DIR)"
	-@echo "Using PETSC_ARCH=$(PETSC_ARCH)"
	-@echo $(ruler)
	 @$(OMAKE) -j $(MAKE_NP) all
	-@echo $(ruler)
.PHONY: all-parallel

OBJDIR := $(PETSCOPT_ARCH)/obj
MODDIR := $(PETSCOPT_ARCH)/include
LIBDIR := $(abspath $(PETSCOPT_ARCH)/lib)

libpetscopt_shared := $(LIBDIR)/libpetscopt.$(SL_LINKER_SUFFIX)
libpetscopt_static := $(LIBDIR)/libpetscopt.$(AR_LIB_SUFFIX)
libpetscopt := $(if $(filter-out no,$(BUILDSHAREDLIB)),$(libpetscopt_shared),$(libpetscopt_static))
alllib : config $(generated) $(libpetscopt)
.PHONY: alllib

all : alllib

ifeq ($(V),)           # Print help and short compile line
  quiet_HELP := "Use \"$(MAKE) V=1\" to see the verbose compile lines.\n"
  quiet = @printf $(quiet_HELP)$(eval quiet_HELP:=)"  %-10s %s\n" "$1$2" "$@"; $($1)
else ifeq ($(V),0)     # Same as previous, but do not print any help
  quiet = @printf "  %-10s %s\n" "$1$2" "$@"; $($1)
else                   # Show the full command line
  quiet = $($1)
endif

pcc = $(if $(findstring CONLY,$(PETSC_LANGUAGE)),CC,CXX)
COMPILE.cc = $(call quiet,$(pcc)) $(PCC_FLAGS) $(CFLAGS) $(CCPPFLAGS) $(C_DEPFLAGS) -c
COMPILE.cpp = $(call quiet,CXX) $(CXX_FLAGS) $(CXXFLAGS) $(CCPPFLAGS) $(CXX_DEPFLAGS) -c
ifneq ($(FC_MODULE_OUTPUT_FLAG),)
COMPILE.fc = $(call quiet,FC) $(FC_FLAGS) $(FFLAGS) $(FCPPFLAGS) $(FC_DEPFLAGS) $(FC_MODULE_OUTPUT_FLAG)$(MODDIR) -c
else
FCMOD = cd $(MODDIR) && $(FC)
COMPILE.fc = $(call quiet,FCMOD) $(FC_FLAGS) $(FFLAGS) $(FCPPFLAGS) $(FC_DEPFLAGS) -c
endif

generated := $(PETSCOPT_DIR)/$(PETSCOPT_ARCH)/lib/petscopt/conf/files

.SECONDEXPANSION: # to expand $$(@D)/.DIR

write-variable = @printf "override $2 = $($2)\n" >> $1;
write-confheader-pre    = @printf "\#if !defined(__PETSCOPTCONF_H)\n\#define __PETSCOPTCONF_H\n" >> $1;
write-confheader-define = printf "\n\#if !defined(PETSCOPT_HAVE_$2)\n\#define PETSCOPT_HAVE_$2 1\n\#endif\n" >> $1;
write-confheader-post = @printf "\n\#endif\n" >> $1;
petscopt-conf-root = $(PETSCOPT_ARCH)
config-petsc      := $(petscopt-conf-root)/lib/petscopt/conf/config-petsc
config-petscopt   := $(petscopt-conf-root)/lib/petscopt/conf/config-petscopt
config-mfem       := $(petscopt-conf-root)/lib/petscopt/conf/config-mfem
config-confheader := $(petscopt-conf-root)/include/petscoptconf.h
$(config-petsc) : | $$(@D)/.DIR
	$(call write-variable,$@,PETSC_DIR)
	$(call write-variable,$@,PETSC_ARCH)
$(config-petscopt) : | $$(@D)/.DIR
	$(call write-variable,$@,CFLAGS)
	$(call write-variable,$@,CXXFLAGS)
$(config-mfem) : | $$(@D)/.DIR
	$(call write-variable,$@,with_mfem)
	$(call write-variable,$@,with_mfem_install)
	$(call write-variable,$@,MFEM_DIR)
$(config-confheader) : | $$(@D)/.DIR
	$(call write-confheader-pre,$@)
	@if [ "${with_mfem}" = "1" ]; then \
	  $(call write-confheader-define,$@,MFEMOPT) \
	fi
	$(call write-confheader-post,$@)

config-petsc : $(config-petsc)
config-petscopt : $(config-petscopt)
config-mfem : $(config-mfem)
config-confheader : config-vars $(config-confheader)

config-vars : config-petsc config-petscopt config-mfem
config : config-confheader
config-clean : clean
	$(RM) $(config-confheader) $(config-petsc) $(config-petscopt) $(config-mfem)
	$(RM) $(generated)
.PHONY:  config-petsc config-petscopt config-mfem config-vars config-confheader config

spkgs := ts,tao,mfemopt
pkgs := ts tao mfemopt
langs := c cpp

$(generated) : $(config-confheader) | $$(@D)/.DIR
	$(PYTHON) $(PETSC_DIR)/config/gmakegen.py --petsc-arch=$(PETSC_ARCH) --pkg-dir=$(PETSCOPT_DIR) --pkg-name=petscopt --pkg-arch=$(PETSCOPT_ARCH) --pkg-pkgs=$(spkgs)

-include $(generated)

concatlang = $(foreach lang, $(langs), $(srcs-$(1).$(lang):src/%.$(lang)=$(OBJDIR)/%.o))
srcs.o := $(foreach pkg, $(pkgs), $(call concatlang,$(pkg)))

objects.o := $(srcs.o)
.SECONDARY: $(objects.o)

$(libpetscopt_static) : objs := $(objects.o)
$(libpetscopt_shared) : objs := $(objects.o)
$(libpetscopt_shared) : libs := $(PETSC_LIB)
$(libpetscopt_shared) : LDSL := CLINKER

%.$(SL_LINKER_SUFFIX) : $$(objs) | $$(@D)/.DIR
	$(call quiet,$(LDSL)) -shared -o $@ $^ $(libs)
ifneq ($(DSYMUTIL),true)
	$(call quiet,DSYMUTIL) $@
endif

%.$(AR_LIB_SUFFIX) : $$(objs) | $$(@D)/.DIR
ifeq ($(findstring win32fe lib,$(AR)),)
	@$(RM) $@
	$(call quiet,AR) $(AR_FLAGS) $@ $^
	$(call quiet,RANLIB) $@
else
	@$(RM) $@ $@.args
	@cygpath -w $^ > $@.args
	$(call quiet,AR) $(AR_FLAGS) $@ @$@.args
	@$(RM) $@.args
endif

$(OBJDIR)/%.o : src/%.c | $$(@D)/.DIR
	$(COMPILE.cc) $(abspath $<) -o $@

$(OBJDIR)/%.o : src/%.cpp | $$(@D)/.DIR
	$(COMPILE.cpp) $(abspath $<) -o $@

$(OBJDIR)/%.o : src/%.F90 | $$(@D)/.DIR $(MODDIR)/.DIR
	$(COMPILE.fc) $(abspath $<) -o $(if $(FCMOD),$(abspath $@),$@)

%/.DIR :
	@$(MKDIR) $(@D)
	@touch $@

.PRECIOUS: %/.DIR
.SUFFIXES: # Clear .SUFFIXES because we don't use implicit rules
.DELETE_ON_ERROR: # Delete likely-corrupt target file if rule fails
.PHONY: all check clean distclean install

#check :
#	@$(OMAKE) -C test/ check

clean :
	$(RM) -r $(OBJDIR) $(LIBDIR)/libpetscopt*.*
distclean :
	@echo "*** Deleting all build files ***"
	-$(RM) -r $(PETSCOPT_DIR)/$(PETSCOPT_ARCH)/

#PREFIX = /tmp/petscopt
#find-install-dir = \
#find $2 -type d -exec \
#install -m $1 -d "$(PREFIX)/{}" \;
#find-install = \
#find $2 -type f -name $3 -exec \
#install -m $1 "{}" "$(PREFIX)/$(if $4,$4,{})" \;
#install :
#	@echo "*** Installing PETSCOPT in PREFIX=$(PREFIX) ***"
#	@$(call find-install-dir,755,bin)
#	@$(call find-install-dir,755,include)
#	@$(call find-install-dir,755,lib)
#	@$(call find-install,755,bin,'petscopt-*')
#	@$(call find-install,644,include,'*.h')
#	@$(call find-install,644,lib/petscopt/app,'*')
#	@$(call find-install,644,lib/petscopt/conf,'*')
#	@$(call find-install,644,lib/petscopt/python,'*.py')
#	@$(call find-install,644,$(PETSCOPT_ARCH)/lib,'libpetscopt*.$(AR_LIB_SUFFIX)',lib)
#	@$(call find-install,755,$(PETSCOPT_ARCH)/lib,'libpetscopt*.$(SL_LINKER_SUFFIX)',lib)
#	@printf "override PETSCOPT_ARCH =\n" > $(PREFIX)/lib/petscopt/conf/arch;
#	@$(OMAKE) petscopt-conf-root=$(PREFIX) config;
#	@$(RM) $(PREFIX)/lib/petscopt/conf/.DIR;

# TAGS generation
alletags :
	-@$(PYTHON) $(PETSC_DIR)/lib/petsc/bin/maint/generateetags.py
deleteetags :
	-@$(RM) CTAGS TAGS
allctags :
	-@ctags -o $(PETSC_DIR)/.vimctags -R --exclude=$(PETSC_DIR)/include/petsc/finclude $(PETSC_DIR)/src/ $(PETSC_DIR)/include/
	-@ctags -o .vimctags -R src/ include/
deletectags :
	-@$(RM) $(PETSC_DIR)/.vimctags
	-@$(RM) .vimctags
.PHONY: alletags deleteetags allctags deletectags

# make print VAR=the-variable
print : ; @echo "$($(VAR))"
.PHONY: print
# make print-VARIABLE
print-% : ; @echo "$* = $($*)"
.PHONY: print-%


objects.d := $(objects.o:%.o=%.d)
# Tell make that objects.d are all up to date. Without
# this, the include below has quadratic complexity.
$(objects.d) : ;
-include $(objects.d)

# Handle test framework (also defines compilation lines for both lib and test)
include ./makefile.test        # This must be below the all target
